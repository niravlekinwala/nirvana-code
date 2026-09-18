use anyhow::{bail, Result};
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc::UnboundedSender;

use crate::hardware::SiliconProfile;

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token(String),
    Stats {
        ttft_ms: u128,
        tokens_per_sec: f64,
        total_tokens: usize,
        prefix_tokens_reused: usize,
        prefix_cache_hit: bool,
        kv_type: String,
        mlock_active: bool,
    },
    Done,
    Error(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvQuantMode {
    Auto,
    Q8_0,
    Q4_0,
    F16,
}

impl KvQuantMode {
    pub fn to_llama_type(self) -> KvCacheType {
        match self {
            KvQuantMode::Auto => KvCacheType::F16,
            KvQuantMode::Q8_0 => KvCacheType::Q8_0,
            KvQuantMode::Q4_0 => KvCacheType::Q4_0,
            KvQuantMode::F16 => KvCacheType::F16,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            KvQuantMode::Auto => "Auto [F16 Peak Speed]",
            KvQuantMode::Q8_0 => "Q8_0 [50% RAM Saved]",
            KvQuantMode::Q4_0 => "Q4_0 [75% RAM Saved]",
            KvQuantMode::F16 => "F16 [Peak Speed 120+ tok/s]",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub min_p: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub use_ngram_speculative: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            temperature: 0.7,
            min_p: 0.05,
            top_p: 0.9,
            top_k: 40,
            use_ngram_speculative: false,
        }
    }
}

pub struct ModelEngine {
    backend: Arc<LlamaBackend>,
    pub model: Arc<LlamaModel>,
    // Persistent context for Prefix Caching across generations
    context: Mutex<Option<LlamaContext<'static>>>,
    cached_tokens: Mutex<Vec<LlamaToken>>,
    pub kv_mode: KvQuantMode,
    pub use_mlock: bool,
    pub n_gpu_layers: u32,
    pub n_ctx: u32,
}

unsafe impl Send for ModelEngine {}
unsafe impl Sync for ModelEngine {}

impl ModelEngine {
    /// Intelligently select optimal KV-cache format based on model parameters & system RAM
    pub fn resolve_auto_kv(model: &LlamaModel) -> KvQuantMode {
        let n_params = model.n_params();
        let sys = SiliconProfile::detect();

        if n_params <= 3_500_000_000 {
            // Models <= 3.5B (e.g. Qwen 1.5B/3B): F16 gives peak 120+ tok/s decode throughput
            KvQuantMode::F16
        } else if n_params < 20_000_000_000 {
            // Models 7B - 14B: Q8_0 halves attention memory bandwidth and VRAM
            KvQuantMode::Q8_0
        } else if sys.memory_gb <= 16 {
            // Models >= 20B on 16GB Macs: Q4_0 cuts KV RAM by 75% so 27B fits without swapping
            KvQuantMode::Q4_0
        } else {
            KvQuantMode::Q8_0
        }
    }

    pub fn load(
        model_path: &Path,
        n_gpu_layers: u32,
        use_mlock: bool,
        kv_mode: KvQuantMode,
        n_ctx: u32,
    ) -> Result<Self> {
        let mut backend = LlamaBackend::init()?;
        backend.void_logs();
        let backend = Arc::new(backend);

        // Apple Silicon Memory Locking (mlock) prevents virtual memory page-outs
        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(n_gpu_layers)
            .with_use_mlock(use_mlock);

        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)?;
        let model = Arc::new(model);

        // Resolve Auto KV-cache mode
        let resolved_kv = if kv_mode == KvQuantMode::Auto {
            Self::resolve_auto_kv(&model)
        } else {
            kv_mode
        };

        Ok(Self {
            backend,
            model,
            context: Mutex::new(None),
            cached_tokens: Mutex::new(Vec::new()),
            kv_mode: resolved_kv,
            use_mlock,
            n_gpu_layers,
            n_ctx,
        })
    }

    pub fn total_layers(&self) -> u32 {
        self.model.n_layer()
    }

    pub fn offloaded_layers(&self) -> u32 {
        self.model.n_layer().min(self.n_gpu_layers)
    }

    /// Explicitly clear the persistent prefix cache
    pub fn clear_cache(&self) {
        if let Ok(mut ctx_guard) = self.context.lock() {
            if let Some(ctx) = ctx_guard.as_mut() {
                ctx.clear_kv_cache();
            }
        }
        if let Ok(mut tokens_guard) = self.cached_tokens.lock() {
            tokens_guard.clear();
        }
    }

    /// Backward-compatible stream generation
    pub fn stream_generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        cancel_token: Arc<AtomicBool>,
        tx: UnboundedSender<StreamEvent>,
    ) -> Result<()> {
        let config = GenerationConfig {
            max_tokens,
            temperature,
            min_p: 0.05,
            top_p: 0.9,
            top_k: 40,
            use_ngram_speculative: false,
        };
        self.stream_generate_with_config(prompt, &config, cancel_token, tx)
    }

    /// Full high-performance stream generation with Min-P, Context Shift, and Prompt Lookup Decoding
    pub fn stream_generate_with_config(
        &self,
        prompt: &str,
        config: &GenerationConfig,
        cancel_token: Arc<AtomicBool>,
        tx: UnboundedSender<StreamEvent>,
    ) -> Result<()> {
        let start_time = Instant::now();

        // 1. Tokenize incoming prompt
        let prompt_tokens = match self.model.str_to_token(prompt, AddBos::Always) {
            Ok(tokens) => tokens,
            Err(e) => {
                let err_msg = format!("Tokenization failed: {}", e);
                let _ = tx.send(StreamEvent::Error(err_msg));
                bail!("Tokenization failed");
            }
        };

        let n_prompt = prompt_tokens.len();
        if n_prompt == 0 {
            let _ = tx.send(StreamEvent::Done);
            return Ok(());
        }

        // 2. Lock and acquire or initialize persistent context
        let mut ctx_guard = self.context.lock().unwrap();
        if ctx_guard.is_none() {
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(Some(NonZeroU32::new(self.n_ctx).unwrap()))
                .with_n_threads(4)
                .with_n_threads_batch(8)
                .with_n_batch(512)
                .with_n_ubatch(512)
                .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_AUTO)
                .with_type_k(self.kv_mode.to_llama_type())
                .with_type_v(self.kv_mode.to_llama_type());

            let ctx = self.model.new_context(&self.backend, ctx_params)?;
            // Safety: model is owned in Arc<LlamaModel> and outlives context
            let static_ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };
            *ctx_guard = Some(static_ctx);
        }

        let ctx = ctx_guard.as_mut().unwrap();

        // 3. Prefix Caching State Verification
        let mut cached_guard = self.cached_tokens.lock().unwrap();
        let mut common_prefix_len = 0;

        // Find longest common prefix between cached tokens and prompt tokens
        while common_prefix_len < cached_guard.len()
            && common_prefix_len < n_prompt
            && cached_guard[common_prefix_len] == prompt_tokens[common_prefix_len]
        {
            common_prefix_len += 1;
        }

        let prefix_cache_hit = common_prefix_len > 0;
        let prefix_tokens_reused = common_prefix_len;

        // Ingest strategy:
        // If common prefix exists: rollback KV cache after common_prefix_len
        // If no common prefix: clear KV cache and start fresh
        if prefix_cache_hit {
            let _ = ctx.kv_cache_seq_rm(0, Some(common_prefix_len as u32), None);
            cached_guard.truncate(common_prefix_len);
        } else {
            ctx.clear_kv_cache();
            cached_guard.clear();
        }

        // Tokens that need evaluation
        let tokens_to_eval = &prompt_tokens[common_prefix_len..];
        let mut n_curr = common_prefix_len;

        // Ingest remaining prompt tokens in chunks
        let mut batch = LlamaBatch::new(2048, 1);
        for chunk in tokens_to_eval.chunks(2048) {
            if cancel_token.load(Ordering::Relaxed) {
                let _ = tx.send(StreamEvent::Done);
                return Ok(());
            }

            batch.clear();
            let chunk_len = chunk.len();
            for (i, &token) in chunk.iter().enumerate() {
                let is_last = (n_curr + i + 1) == n_prompt;
                batch.add(token, (n_curr + i) as i32, &[0], is_last)?;
            }
            n_curr += chunk_len;
            ctx.decode(&mut batch)?;
        }

        // Update cached tokens to reflect full prompt in KV cache
        cached_guard.extend_from_slice(tokens_to_eval);

        // 4. Setup Sampler with Min-P filtering
        let mut sampler = if config.temperature <= 0.05 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::min_p(config.min_p, 1),
                LlamaSampler::top_k(config.top_k),
                LlamaSampler::top_p(config.top_p, 1),
                LlamaSampler::temp(config.temperature),
                LlamaSampler::dist(42),
            ])
        };

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut total_generated = 0;

        // Sample initial token from the prompt evaluation
        let mut current_token = sampler.sample(ctx, batch.n_tokens() - 1);
        sampler.accept(current_token);

        if self.model.is_eog_token(current_token) {
            let _ = tx.send(StreamEvent::Done);
            return Ok(());
        }

        let first_token_time = Some(Instant::now());
        total_generated += 1;

        if let Ok(piece) = self.model.token_to_piece(current_token, &mut decoder, false, None) {
            if !piece.is_empty() {
                let _ = tx.send(StreamEvent::Token(piece));
            }
        }
        cached_guard.push(current_token);

        // 5. Autoregressive / Speculative Generation Loop
        while total_generated < config.max_tokens {
            if cancel_token.load(Ordering::Relaxed) {
                break;
            }

            // Context Shifting: Prevent overflow when approaching context capacity
            if (n_curr + 32) >= self.n_ctx as usize {
                let keep = (self.n_ctx / 8).max(32) as u32;
                let drop = (self.n_ctx / 4).max(64) as u32;
                if (n_curr as u32) > keep + drop {
                    let _ = ctx.kv_cache_seq_rm(0, Some(keep), Some(keep + drop));
                    let _ = ctx.kv_cache_seq_add(0, Some(keep + drop), Some(n_curr as u32), -(drop as i32));
                    n_curr -= drop as usize;
                    if cached_guard.len() > (keep + drop) as usize {
                        cached_guard.drain((keep as usize)..(keep + drop) as usize);
                    }
                }
            }

            // Prompt Lookup Decoding: Check for repeating N-grams in context
            let mut ngram_candidates: Option<Vec<LlamaToken>> = None;
            if config.use_ngram_speculative && cached_guard.len() >= 6 {
                let ngram_len = 3;
                let draft_len = 3;
                let query = &cached_guard[cached_guard.len() - ngram_len..];
                let search_limit = cached_guard.len() - ngram_len;
                for i in (0..search_limit).rev() {
                    if &cached_guard[i..i + ngram_len] == query {
                        let start = i + ngram_len;
                        let end = (start + draft_len).min(search_limit);
                        if end > start {
                            ngram_candidates = Some(cached_guard[start..end].to_vec());
                            break;
                        }
                    }
                }
            }

            if let Some(cands) = ngram_candidates {
                // Batch-evaluate current token and all speculative candidates simultaneously
                batch.clear();
                batch.add(current_token, n_curr as i32, &[0], true)?;
                for (k, &cand) in cands.iter().enumerate() {
                    batch.add(cand, (n_curr + 1 + k) as i32, &[0], true)?;
                }
                ctx.decode(&mut batch)?;

                // Verification pass
                let mut verified = 0;
                let mut next_token = sampler.sample(ctx, 0);
                sampler.accept(next_token);

                for (k, &cand) in cands.iter().enumerate() {
                    if next_token == cand {
                        verified += 1;
                        total_generated += 1;
                        cached_guard.push(cand);
                        if let Ok(piece) = self.model.token_to_piece(cand, &mut decoder, false, None) {
                            if !piece.is_empty() {
                                let _ = tx.send(StreamEvent::Token(piece));
                            }
                        }
                        if self.model.is_eog_token(cand) || total_generated >= config.max_tokens {
                            break;
                        }
                        next_token = sampler.sample(ctx, (k + 1) as i32);
                        sampler.accept(next_token);
                    } else {
                        break;
                    }
                }

                // If candidate mismatch occurred, accept next_token (the true model prediction)
                if verified < cands.len() && !self.model.is_eog_token(current_token) && total_generated < config.max_tokens {
                    total_generated += 1;
                    cached_guard.push(next_token);
                    if let Ok(piece) = self.model.token_to_piece(next_token, &mut decoder, false, None) {
                        if !piece.is_empty() {
                            let _ = tx.send(StreamEvent::Token(piece));
                        }
                    }
                }

                // Rollback any unverified speculative tokens from the KV cache
                let valid_pos = n_curr + 1 + verified;
                let _ = ctx.kv_cache_seq_rm(0, Some(valid_pos as u32), None);
                n_curr = valid_pos;
                current_token = next_token;

                if self.model.is_eog_token(current_token) {
                    break;
                }
            } else {
                // Standard Autoregressive Single-Token Step
                batch.clear();
                batch.add(current_token, n_curr as i32, &[0], true)?;
                n_curr += 1;
                ctx.decode(&mut batch)?;

                current_token = sampler.sample(ctx, 0);
                sampler.accept(current_token);

                if self.model.is_eog_token(current_token) {
                    break;
                }

                total_generated += 1;
                if let Ok(piece) = self.model.token_to_piece(current_token, &mut decoder, false, None) {
                    if !piece.is_empty() {
                        let _ = tx.send(StreamEvent::Token(piece));
                    }
                }
                cached_guard.push(current_token);
            }
        }

        let elapsed = start_time.elapsed();
        let ttft_ms = first_token_time
            .map(|t| t.duration_since(start_time).as_millis())
            .unwrap_or(elapsed.as_millis());

        let decode_secs = first_token_time
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.001);

        let tps = if total_generated > 1 && decode_secs > 0.0 {
            (total_generated - 1) as f64 / decode_secs
        } else {
            0.0
        };

        let _ = tx.send(StreamEvent::Stats {
            ttft_ms,
            tokens_per_sec: tps,
            total_tokens: total_generated,
            prefix_tokens_reused,
            prefix_cache_hit,
            kv_type: self.kv_mode.label().to_string(),
            mlock_active: self.use_mlock,
        });

        let _ = tx.send(StreamEvent::Done);
        Ok(())
    }
}
