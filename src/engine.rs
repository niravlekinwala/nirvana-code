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
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;
use tokio::sync::mpsc::UnboundedSender;

use crate::hardware::SiliconProfile;
use crate::mlx_engine::MlxEngine;
use crate::model_manager::ModelManager;

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token(String),
    Stats {
        ttft_ms: u128,
        tokens_per_sec: f64,
        total_tokens: usize,
        prompt_tokens: usize,
        context_used: usize,
        context_capacity: u32,
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
    /// Sampler RNG seed. `None` draws a fresh seed per generation so regenerating
    /// the same prompt gives a different answer.
    pub seed: Option<u32>,
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
            seed: None,
        }
    }
}

impl GenerationConfig {
    fn resolve_seed(&self) -> u32 {
        self.seed.unwrap_or_else(|| {
            use std::hash::{BuildHasher, Hasher};
            std::collections::hash_map::RandomState::new().build_hasher().finish() as u32
        })
    }
}

/// Build the sampler chain for a generation. Greedy below a tiny temperature,
/// otherwise llama.cpp's default order (top-k → top-p → min-p → temperature).
pub(crate) fn build_sampler(config: &GenerationConfig) -> LlamaSampler {
    if config.temperature <= 0.05 {
        LlamaSampler::greedy()
    } else {
        LlamaSampler::chain_simple([
            LlamaSampler::top_k(config.top_k),
            LlamaSampler::top_p(config.top_p, 1),
            LlamaSampler::min_p(config.min_p, 1),
            LlamaSampler::temp(config.temperature),
            LlamaSampler::dist(config.resolve_seed()),
        ])
    }
}

/// Pin the calling thread to User-Interactive QoS so macOS schedules the
/// decode loop (and its Metal command encoding) on performance cores.
pub(crate) fn boost_thread_qos() {
    #[cfg(target_os = "macos")]
    unsafe {
        unsafe extern "C" {
            fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
        }
        let _ = pthread_set_qos_class_self_np(0x21, 0);
    }
}

/// Mirror of the tokens resident in sequence 0 of a context's KV cache, in
/// position order. Invariant: `tokens[i]` is exactly what was decoded at
/// position `i`; a sampled-but-not-yet-decoded token is never stored here.
/// Shared by the GGUF and speculative engines so prefix reuse behaves identically.
#[derive(Default)]
pub(crate) struct PrefixCache {
    pub tokens: Vec<LlamaToken>,
}

impl PrefixCache {
    /// Roll the KV cache back to the longest prefix shared with `prompt` and
    /// return how many tokens were reused. Always strictly less than
    /// `prompt.len()`: at least one prompt token is re-evaluated so the logits
    /// used for the first sample are fresh (a full hit would otherwise sample
    /// from whatever the previous generation left behind).
    pub(crate) fn sync(&mut self, ctx: &mut LlamaContext, prompt: &[LlamaToken]) -> usize {
        let common = reusable_prefix_len(&self.tokens, prompt);
        self.rollback(ctx, common);
        common
    }

    /// Drop everything at position >= `keep` from the KV cache and the mirror.
    /// The KV is always trimmed, even when the mirror is already short: after
    /// speculative verification the KV holds rejected draft tokens the mirror
    /// never recorded.
    pub(crate) fn rollback(&mut self, ctx: &mut LlamaContext, keep: usize) {
        if keep == 0 {
            ctx.clear_kv_cache();
        } else {
            let _ = ctx.kv_cache_seq_rm(0, Some(keep as u32), None);
        }
        self.tokens.truncate(keep);
    }

    /// Decode `tokens` starting at the current end of the cache, in chunks no
    /// larger than the batch, requesting logits only for the final token.
    /// Returns `false` if cancelled part-way (the cache stays consistent).
    pub(crate) fn prefill(
        &mut self,
        ctx: &mut LlamaContext,
        batch: &mut LlamaBatch,
        tokens: &[LlamaToken],
        batch_size: usize,
        cancel: &AtomicBool,
    ) -> Result<bool> {
        let n_total = tokens.len();
        let mut done = 0;
        for chunk in tokens.chunks(batch_size) {
            if cancel.load(Ordering::Relaxed) {
                return Ok(false);
            }
            batch.clear();
            let base = self.tokens.len();
            for (i, &token) in chunk.iter().enumerate() {
                let is_last = done + i + 1 == n_total;
                batch.add(token, (base + i) as i32, &[0], is_last)?;
            }
            ctx.decode(batch)?;
            self.tokens.extend_from_slice(chunk);
            done += chunk.len();
        }
        Ok(true)
    }
}

/// Length of the prefix of `prompt` already resident in `cached`, capped at
/// `prompt.len() - 1` so the final prompt token is always re-evaluated.
pub(crate) fn reusable_prefix_len(cached: &[LlamaToken], prompt: &[LlamaToken]) -> usize {
    let limit = prompt.len().saturating_sub(1);
    cached
        .iter()
        .zip(prompt)
        .take(limit)
        .take_while(|(a, b)| a == b)
        .count()
}

/// Prompt-lookup drafting: find the most recent earlier occurrence of the
/// last `n` tokens of `history ++ [pending]` and propose the up-to-`k` tokens
/// that followed it.
pub(crate) fn ngram_draft(
    history: &[LlamaToken],
    pending: LlamaToken,
    n: usize,
    k: usize,
) -> Option<Vec<LlamaToken>> {
    let len = history.len() + 1;
    if len < n + 1 {
        return None;
    }
    let at = |i: usize| if i < history.len() { history[i] } else { pending };
    let query_start = len - n;
    for i in (0..query_start).rev() {
        if (0..n).all(|j| at(i + j) == at(query_start + j)) {
            let start = i + n;
            let end = (start + k).min(len);
            if end > start {
                return Some((start..end).map(at).collect());
            }
        }
    }
    None
}

static BACKEND: OnceLock<Arc<SharedBackend>> = OnceLock::new();

pub struct SharedBackend(pub LlamaBackend);
unsafe impl Send for SharedBackend {}
unsafe impl Sync for SharedBackend {}

impl std::ops::Deref for SharedBackend {
    type Target = LlamaBackend;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl SharedBackend {
    pub fn get() -> Result<Arc<Self>> {
        if let Some(b) = BACKEND.get() {
            return Ok(b.clone());
        }
        let mut backend = LlamaBackend::init()?;
        backend.void_logs();
        let shared = Arc::new(SharedBackend(backend));
        let _ = BACKEND.set(shared.clone());
        Ok(shared)
    }
}

pub struct ModelEngine {
    // Persistent context for prefix caching across generations. Declared
    // before `model`: Rust drops fields in declaration order and the context
    // borrows the model (see the transmute in `stream_generate_with_config`).
    context: Mutex<Option<LlamaContext<'static>>>,
    cached_tokens: Mutex<PrefixCache>,
    backend: Arc<SharedBackend>,
    pub model: Arc<LlamaModel>,
    pub kv_mode: KvQuantMode,
    pub use_mlock: bool,
    pub n_gpu_layers: u32,
    pub n_ctx: AtomicU32,
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
        let backend = SharedBackend::get()?;

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
            cached_tokens: Mutex::new(PrefixCache::default()),
            kv_mode: resolved_kv,
            use_mlock,
            n_gpu_layers,
            n_ctx: AtomicU32::new(n_ctx),
        })
    }

    pub fn total_layers(&self) -> u32 {
        self.model.n_layer()
    }

    pub fn offloaded_layers(&self) -> u32 {
        self.model.n_layer().min(self.n_gpu_layers)
    }

    /// Dynamically reconfigure context size without reloading model weights
    pub fn reconfigure_context(&self, new_n_ctx: u32) -> Result<()> {
        let mut ctx_guard = self.context.lock().unwrap();
        *ctx_guard = None;
        self.cached_tokens.lock().unwrap().tokens.clear();
        self.n_ctx.store(new_n_ctx, Ordering::Relaxed);
        Ok(())
    }

    /// Explicitly clear the persistent prefix cache
    pub fn clear_cache(&self) {
        if let Ok(mut ctx_guard) = self.context.lock() {
            if let Some(ctx) = ctx_guard.as_mut() {
                ctx.clear_kv_cache();
            }
        }
        if let Ok(mut cache) = self.cached_tokens.lock() {
            cache.tokens.clear();
        }
    }

    /// Backward-compatible stream generation
    #[allow(dead_code)]
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
            seed: None,
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
                let err_msg = format!("Tokenization failed: {e}");
                let _ = tx.send(StreamEvent::Error(err_msg));
                bail!("Tokenization failed");
            }
        };

        let max_ctx = self.n_ctx.load(Ordering::Relaxed) as usize;
        let safe_prompt_limit = max_ctx.saturating_sub(config.max_tokens.min(max_ctx / 2).max(128));

        let prompt_tokens = if prompt_tokens.len() > safe_prompt_limit {
            // Gracefully truncate prompt if it exceeds context capacity: keep system header and tail
            let head_len = 256.min(safe_prompt_limit / 4);
            let tail_len = safe_prompt_limit.saturating_sub(head_len);
            let mut truncated = Vec::with_capacity(safe_prompt_limit);
            truncated.extend_from_slice(&prompt_tokens[..head_len]);
            truncated.extend_from_slice(&prompt_tokens[prompt_tokens.len().saturating_sub(tail_len)..]);
            truncated
        } else {
            prompt_tokens
        };

        let n_prompt = prompt_tokens.len();
        if n_prompt == 0 {
            let _ = tx.send(StreamEvent::Done);
            return Ok(());
        }

        boost_thread_qos();

        let n_ctx_val = self.n_ctx.load(Ordering::Relaxed);
        let n_batch = 2048.min(n_ctx_val);
        let n_ubatch = 512.min(n_batch);
        let p_cores = SiliconProfile::detect().p_cores.max(1) as i32;

        // 2. Lock and acquire or initialize persistent context
        let mut ctx_guard = self.context.lock().unwrap();
        if ctx_guard.is_none() {
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(Some(NonZeroU32::new(n_ctx_val).unwrap()))
                .with_n_threads(p_cores)
                .with_n_threads_batch(p_cores)
                .with_n_batch(n_batch)
                .with_n_ubatch(n_ubatch)
                .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED)
                .with_type_k(self.kv_mode.to_llama_type())
                .with_type_v(self.kv_mode.to_llama_type());

            let ctx = self.model.new_context(&self.backend, ctx_params)?;
            // Safety: model is owned in Arc<LlamaModel> and outlives context
            let static_ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };
            *ctx_guard = Some(static_ctx);
        }

        let ctx = ctx_guard.as_mut().unwrap();

        // 3. Prefix cache: roll back to the shared prefix, then evaluate the rest
        let mut cache = self.cached_tokens.lock().unwrap();
        let prefix_tokens_reused = cache.sync(ctx, &prompt_tokens);
        let prefix_cache_hit = prefix_tokens_reused > 0;

        let batch_size = (n_batch as usize).min(512);
        let mut batch = LlamaBatch::new(batch_size, 1);
        if !cache.prefill(ctx, &mut batch, &prompt_tokens[prefix_tokens_reused..], batch_size, &cancel_token)? {
            let _ = tx.send(StreamEvent::Done);
            return Ok(());
        }

        // 4. Sampler
        let mut sampler = build_sampler(config);
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut total_generated = 0;

        // `current_token` is the pending token: sampled and emitted, but not yet
        // decoded into the KV. `cache.tokens` therefore never includes it.
        let mut current_token = sampler.sample(ctx, batch.n_tokens() - 1);
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

        // 5. Autoregressive / prompt-lookup generation loop
        const NGRAM_LEN: usize = 3;
        const NGRAM_DRAFT: usize = 3;
        while total_generated < config.max_tokens {
            if cancel_token.load(Ordering::Relaxed) {
                break;
            }

            let n_past = cache.tokens.len();

            // Context shifting: drop a block after the head and slide the rest down
            if (n_past + 32) >= n_ctx_val as usize {
                let keep = (n_ctx_val / 8).max(32) as usize;
                let drop = (n_ctx_val / 4).max(64) as usize;
                if n_past > keep + drop {
                    let _ = ctx.kv_cache_seq_rm(0, Some(keep as u32), Some((keep + drop) as u32));
                    let _ = ctx.kv_cache_seq_add(0, Some((keep + drop) as u32), Some(n_past as u32), -(drop as i32));
                    cache.tokens.drain(keep..keep + drop);
                }
            }
            let n_past = cache.tokens.len();

            let cands = if config.use_ngram_speculative {
                ngram_draft(&cache.tokens, current_token, NGRAM_LEN, NGRAM_DRAFT)
            } else {
                None
            };

            if let Some(cands) = cands {
                // Evaluate the pending token and the drafted continuation in one pass.
                // Logits at index i predict the token after batch[i], i.e. cands[i].
                batch.clear();
                batch.add(current_token, n_past as i32, &[0], true)?;
                for (k, &cand) in cands.iter().enumerate() {
                    batch.add(cand, (n_past + 1 + k) as i32, &[0], true)?;
                }
                ctx.decode(&mut batch)?;
                cache.tokens.push(current_token);

                let mut finished = false;
                let mut next_token = current_token;
                for i in 0..=cands.len() {
                    next_token = sampler.sample(ctx, i as i32);
                    if self.model.is_eog_token(next_token) {
                        finished = true;
                        break;
                    }
                    if i < cands.len() && next_token == cands[i] {
                        cache.tokens.push(cands[i]);
                        total_generated += 1;
                        if let Ok(piece) = self.model.token_to_piece(cands[i], &mut decoder, false, None) {
                            if !piece.is_empty() {
                                let _ = tx.send(StreamEvent::Token(piece));
                            }
                        }
                        if total_generated >= config.max_tokens {
                            finished = true;
                            break;
                        }
                    } else {
                        break;
                    }
                }

                // Evict the drafted tokens that were not accepted
                let keep = cache.tokens.len();
                cache.rollback(ctx, keep);
                if finished {
                    break;
                }

                // The target's own sample is always a real output token
                current_token = next_token;
                total_generated += 1;
                if let Ok(piece) = self.model.token_to_piece(current_token, &mut decoder, false, None) {
                    if !piece.is_empty() {
                        let _ = tx.send(StreamEvent::Token(piece));
                    }
                }
            } else {
                // Standard single-token step
                batch.clear();
                batch.add(current_token, n_past as i32, &[0], true)?;
                ctx.decode(&mut batch)?;
                cache.tokens.push(current_token);

                current_token = sampler.sample(ctx, 0);
                if self.model.is_eog_token(current_token) {
                    break;
                }

                total_generated += 1;
                if let Ok(piece) = self.model.token_to_piece(current_token, &mut decoder, false, None) {
                    if !piece.is_empty() {
                        let _ = tx.send(StreamEvent::Token(piece));
                    }
                }
            }
        }

        let context_used = cache.tokens.len();
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
            prompt_tokens: n_prompt,
            context_used,
            context_capacity: n_ctx_val,
            prefix_tokens_reused,
            prefix_cache_hit,
            kv_type: self.kv_mode.label().to_string(),
            mlock_active: self.use_mlock,
        });

        let _ = tx.send(StreamEvent::Done);
        Ok(())
    }
}

#[derive(Clone)]
pub enum InferenceEngine {
    Gguf(Arc<ModelEngine>),
    Mlx(Arc<MlxEngine>),
}

impl InferenceEngine {
    pub fn load(
        model_path: &Path,
        gpu_layers: u32,
        use_mlock: bool,
        kv_mode: KvQuantMode,
        ctx_size: u32,
    ) -> Result<Self> {
        if ModelManager::is_mlx_model(model_path) {
            let mlx = MlxEngine::load(model_path)?;
            Ok(InferenceEngine::Mlx(Arc::new(mlx)))
        } else {
            let gguf = ModelEngine::load(model_path, gpu_layers, use_mlock, kv_mode, ctx_size)?;
            Ok(InferenceEngine::Gguf(Arc::new(gguf)))
        }
    }

    pub fn stream_generate_with_config(
        &self,
        prompt: &str,
        config: &GenerationConfig,
        cancel_token: Arc<AtomicBool>,
        tx: UnboundedSender<StreamEvent>,
    ) -> Result<()> {
        match self {
            InferenceEngine::Gguf(e) => e.stream_generate_with_config(prompt, config, cancel_token, tx),
            InferenceEngine::Mlx(e) => e.stream_generate_with_config(prompt, config, cancel_token, tx),
        }
    }

    pub fn clear_cache(&self) {
        match self {
            InferenceEngine::Gguf(e) => e.clear_cache(),
            InferenceEngine::Mlx(e) => e.clear_cache(),
        }
    }

    pub fn backend_name(&self) -> &'static str {
        match self {
            InferenceEngine::Gguf(_) => "Metal GGUF",
            InferenceEngine::Mlx(_) => "Apple MLX",
        }
    }

    pub fn kv_label(&self) -> String {
        match self {
            InferenceEngine::Gguf(e) => e.kv_mode.label().to_string(),
            InferenceEngine::Mlx(_) => "Unified LPDDR5 (Apple MLX)".to_string(),
        }
    }

    #[allow(dead_code)]
    pub fn is_mlx(&self) -> bool {
        matches!(self, InferenceEngine::Mlx(_))
    }

    pub fn total_layers(&self) -> u32 {
        match self {
            InferenceEngine::Gguf(e) => e.total_layers(),
            InferenceEngine::Mlx(_) => 32,
        }
    }

    pub fn offloaded_layers(&self) -> u32 {
        match self {
            InferenceEngine::Gguf(e) => e.offloaded_layers(),
            InferenceEngine::Mlx(_) => 32,
        }
    }

    pub fn n_gpu_layers(&self) -> u32 {
        match self {
            InferenceEngine::Gguf(e) => e.n_gpu_layers,
            InferenceEngine::Mlx(_) => 32,
        }
    }

    pub fn use_mlock(&self) -> bool {
        match self {
            InferenceEngine::Gguf(e) => e.use_mlock,
            InferenceEngine::Mlx(_) => true,
        }
    }

    pub fn kv_mode(&self) -> KvQuantMode {
        match self {
            InferenceEngine::Gguf(e) => e.kv_mode,
            InferenceEngine::Mlx(_) => KvQuantMode::Auto,
        }
    }

    pub fn reconfigure_context(&self, new_n_ctx: u32) -> Result<()> {
        match self {
            InferenceEngine::Gguf(e) => e.reconfigure_context(new_n_ctx),
            InferenceEngine::Mlx(_) => Ok(()),
        }
    }

    pub fn n_ctx(&self) -> u32 {
        match self {
            InferenceEngine::Gguf(e) => e.n_ctx.load(Ordering::Relaxed),
            InferenceEngine::Mlx(_) => 32768,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn t(ids: &[i32]) -> Vec<LlamaToken> {
        ids.iter().map(|&i| LlamaToken(i)).collect()
    }

    #[test]
    fn prefix_partial_match() {
        assert_eq!(reusable_prefix_len(&t(&[1, 2, 3, 4]), &t(&[1, 2, 9, 9, 9])), 2);
    }

    #[test]
    fn prefix_no_match() {
        assert_eq!(reusable_prefix_len(&t(&[5, 6]), &t(&[1, 2, 3])), 0);
        assert_eq!(reusable_prefix_len(&[], &t(&[1, 2, 3])), 0);
    }

    #[test]
    fn prefix_full_hit_leaves_last_token_for_reeval() {
        // Regenerating the identical prompt must still decode one token so the
        // logits are fresh, not whatever the previous generation left behind.
        assert_eq!(reusable_prefix_len(&t(&[1, 2, 3]), &t(&[1, 2, 3])), 2);
        // Prompt is a prefix of the cache (e.g. user deleted their last turn)
        assert_eq!(reusable_prefix_len(&t(&[1, 2, 3, 4, 5]), &t(&[1, 2, 3])), 2);
        assert_eq!(reusable_prefix_len(&t(&[7]), &t(&[7])), 0);
    }

    #[test]
    fn ngram_finds_most_recent_continuation() {
        // history ++ pending = [1 2 3 4 5 | 1 2 3 4 6 | 1 2] + 3
        let hist = t(&[1, 2, 3, 4, 5, 1, 2, 3, 4, 6, 1, 2]);
        let draft = ngram_draft(&hist, LlamaToken(3), 3, 3).unwrap();
        // Most recent earlier "1 2 3" is at index 5, followed by 4 6 1
        assert_eq!(draft, t(&[4, 6, 1]));
    }

    #[test]
    fn ngram_draft_is_truncated_at_sequence_end() {
        // history ++ pending = [1 2 3 1 2 3]; the match at index 0 is followed
        // by [1 2 3] and then the sequence ends, so the draft is capped there.
        let hist = t(&[1, 2, 3, 1, 2]);
        let draft = ngram_draft(&hist, LlamaToken(3), 3, 8).unwrap();
        assert_eq!(draft, t(&[1, 2, 3]));
    }

    #[test]
    fn ngram_none_when_no_repeat_or_too_short() {
        assert!(ngram_draft(&t(&[1, 2, 3, 4]), LlamaToken(5), 3, 3).is_none());
        assert!(ngram_draft(&t(&[1, 2]), LlamaToken(3), 3, 3).is_none());
    }
}
