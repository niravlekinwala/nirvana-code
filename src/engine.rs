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
    Q8_0,
    Q4_0,
    F16,
}

impl KvQuantMode {
    pub fn to_llama_type(self) -> KvCacheType {
        match self {
            KvQuantMode::Q8_0 => KvCacheType::Q8_0,
            KvQuantMode::Q4_0 => KvCacheType::Q4_0,
            KvQuantMode::F16 => KvCacheType::F16,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            KvQuantMode::Q8_0 => "Q8_0 [50% RAM Saved]",
            KvQuantMode::Q4_0 => "Q4_0 [75% RAM Saved]",
            KvQuantMode::F16 => "F16 [Standard]",
        }
    }
}

pub struct ModelEngine {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
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

        Ok(Self {
            backend,
            model,
            context: Mutex::new(None),
            cached_tokens: Mutex::new(Vec::new()),
            kv_mode,
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

    pub fn stream_generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
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
            // Remove all KV cache entries strictly from common_prefix_len onwards
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

        // 4. Setup Sampler
        let mut sampler = if temperature <= 0.05 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::top_k(40),
                LlamaSampler::top_p(0.9, 1),
                LlamaSampler::temp(temperature),
                LlamaSampler::dist(42),
            ])
        };

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut first_token_time: Option<Instant> = None;
        let mut total_generated = 0;

        // 5. Autoregressive Generation Loop
        for _ in 0..max_tokens {
            if cancel_token.load(Ordering::Relaxed) {
                break;
            }

            let token = sampler.sample(ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                break;
            }

            if first_token_time.is_none() {
                first_token_time = Some(Instant::now());
            }

            total_generated += 1;

            if let Ok(piece) = self.model.token_to_piece(token, &mut decoder, false, None) {
                if !piece.is_empty() {
                    let _ = tx.send(StreamEvent::Token(piece));
                }
            }

            // Append generated token to cached tokens history
            cached_guard.push(token);

            // Decode the newly generated token for next step
            batch.clear();
            batch.add(token, n_curr as i32, &[0], true)?;
            n_curr += 1;
            ctx.decode(&mut batch)?;
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
