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
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;
use tokio::sync::mpsc::UnboundedSender;

use crate::chat::{ChatMessage, ChatRenderer};
use crate::speculative::SpeculativeEngine;
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
    /// Classic repetition penalty over the last `penalty_last_n` tokens; 1.0 = off.
    pub repeat_penalty: f32,
    /// OpenAI-style frequency / presence penalties; 0.0 = off.
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub penalty_last_n: i32,
    /// DRY (don't repeat yourself) multiplier; 0.0 = off.
    pub dry_multiplier: f32,
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
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            penalty_last_n: 64,
            dry_multiplier: 0.0,
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

/// Build the sampler chain for a generation: optional repetition/DRY
/// penalties first, then greedy below a tiny temperature, otherwise
/// llama.cpp's default order (top-k → top-p → min-p → temperature).
pub(crate) fn build_sampler(model: &LlamaModel, config: &GenerationConfig) -> LlamaSampler {
    let mut chain: Vec<LlamaSampler> = Vec::with_capacity(7);
    let penalties_on = (config.repeat_penalty - 1.0).abs() > f32::EPSILON
        || config.frequency_penalty.abs() > f32::EPSILON
        || config.presence_penalty.abs() > f32::EPSILON;
    if penalties_on {
        chain.push(LlamaSampler::penalties(
            model.n_vocab(),
            config.penalty_last_n,
            config.repeat_penalty,
            config.frequency_penalty,
            config.presence_penalty,
        ));
    }
    if config.dry_multiplier > 0.0 {
        chain.push(LlamaSampler::dry(
            model,
            config.dry_multiplier,
            1.75,
            2,
            config.penalty_last_n,
            ["\n", ":", "\"", "*"],
        ));
    }
    if config.temperature <= 0.05 {
        chain.push(LlamaSampler::greedy());
    } else {
        chain.extend([
            LlamaSampler::top_k(config.top_k),
            LlamaSampler::top_p(config.top_p, 1),
            LlamaSampler::min_p(config.min_p, 1),
            LlamaSampler::temp(config.temperature),
            LlamaSampler::dist(config.resolve_seed()),
        ]);
    }
    LlamaSampler::chain_simple(chain)
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
        if self.rollback(ctx, common) {
            common
        } else {
            // The memory refused a partial removal (e.g. an SWA window that
            // no longer holds those positions): start over.
            self.rollback(ctx, 0);
            0
        }
    }

    /// Drop everything at position >= `keep` from the KV cache and the mirror.
    /// The KV is always trimmed, even when the mirror is already short: after
    /// speculative verification the KV holds rejected draft tokens the mirror
    /// never recorded.
    /// Returns `false` if the KV refused the partial removal; the mirror is
    /// still truncated so the caller can decide how to recover.
    pub(crate) fn rollback(&mut self, ctx: &mut LlamaContext, keep: usize) -> bool {
        let ok = if keep == 0 {
            ctx.clear_kv_cache();
            true
        } else {
            ctx.kv_cache_seq_rm(0, Some(keep as u32), None).is_ok()
        };
        self.tokens.truncate(keep);
        ok
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

static VERBOSE: AtomicBool = AtomicBool::new(false);

/// Keep llama.cpp's own diagnostics on stderr (`--verbose`). Must be called
/// before the backend is first initialised.
pub fn set_verbose(on: bool) {
    VERBOSE.store(on, Ordering::Relaxed);
}

pub fn verbose() -> bool {
    VERBOSE.load(Ordering::Relaxed)
}

impl SharedBackend {
    pub fn get() -> Result<Arc<Self>> {
        if let Some(b) = BACKEND.get() {
            return Ok(b.clone());
        }
        let mut backend = LlamaBackend::init()?;
        if !verbose() {
            backend.void_logs();
        }
        let shared = Arc::new(SharedBackend(backend));
        let _ = BACKEND.set(shared.clone());
        Ok(shared)
    }
}

/// Decides mlock and reports whether the weights fit the GPU-wired budget.
pub(crate) struct MemoryPlan {
    pub model_bytes: u64,
    pub use_mlock: bool,
    pub mlock_disabled: bool,
    pub wired_advice_mb: Option<u64>,
}

impl MemoryPlan {
    pub fn for_model(model_path: &Path, want_mlock: bool) -> Self {
        let model_bytes = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);
        let hw = SiliconProfile::detect();
        // mlock pins every page; above ~70 % of RAM that starves the OS and the
        // KV cache, and the call itself fails noisily on macOS.
        let too_big_to_pin = model_bytes > hw.memory_bytes / 10 * 7;
        let use_mlock = want_mlock && !too_big_to_pin;
        Self {
            model_bytes,
            use_mlock,
            mlock_disabled: want_mlock && !use_mlock,
            wired_advice_mb: hw.suggested_wired_limit_mb(model_bytes),
        }
    }

    pub fn report(&self) {
        if self.mlock_disabled {
            eprintln!(
                "   Memory:      mlock disabled — {:.1} GB of weights exceeds 70% of RAM",
                self.model_bytes as f64 / (1u64 << 30) as f64
            );
        }
        if let Some(mb) = self.wired_advice_mb {
            let hw = SiliconProfile::detect();
            eprintln!(
                "   Memory:      weights ({:.1} GB) exceed the GPU-wired budget ({:.1} GB). To keep the whole model on the GPU run:\n                sudo sysctl iogpu.wired_limit_mb={}",
                self.model_bytes as f64 / (1u64 << 30) as f64,
                hw.gpu_wired_limit_bytes as f64 / (1u64 << 30) as f64,
                mb
            );
        }
    }
}

/// Load weights, letting llama.cpp's fitter pick the offload split when the
/// caller asked for "all layers" (99+). Pinned layer counts are honoured as-is.
pub(crate) fn load_model_fitted(
    backend: &SharedBackend,
    model_path: &Path,
    n_gpu_layers: u32,
    use_mlock: bool,
    n_ctx: u32,
) -> Result<LlamaModel> {
    if n_gpu_layers >= 99 {
        if let Some(cpath) = model_path.to_str().and_then(|s| std::ffi::CString::new(s).ok()) {
            let mut params = Box::pin(LlamaModelParams::default().with_use_mlock(use_mlock));
            let mut cparams = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(n_ctx))
                .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED);
            let n_dev = unsafe { llama_cpp_sys_2::llama_max_devices() };
            let mut margins = vec![1usize << 30; n_dev];
            let log_level = if verbose() {
                llama_cpp_sys_2::GGML_LOG_LEVEL_INFO
            } else {
                llama_cpp_sys_2::GGML_LOG_LEVEL_ERROR
            };
            match params.as_mut().fit_params(&cpath, &mut cparams, &mut margins, 512, log_level) {
                Ok(_) => {
                    let overrides = params.tensor_buft_override_patterns();
                    // -1 means "all layers"; anything else means the fitter cut back
                    if params.n_gpu_layers() >= 0 || !overrides.is_empty() {
                        eprintln!(
                            "   Memory:      auto-fit → {} GPU layers{}",
                            params.n_gpu_layers(),
                            if overrides.is_empty() { String::new() } else { format!(", {} tensors on CPU", overrides.len()) }
                        );
                    }
                    return Ok(LlamaModel::load_from_file(backend, model_path, &params)?);
                }
                Err(_) => {
                    eprintln!("   Memory:      auto-fit found no allocation that fits; loading anyway");
                }
            }
        }
    }
    let params = LlamaModelParams::default()
        .with_n_gpu_layers(n_gpu_layers)
        .with_use_mlock(use_mlock);
    Ok(LlamaModel::load_from_file(backend, model_path, &params)?)
}

const SESSION_MAX_TOKENS: usize = 1024;

static UBATCH: AtomicU32 = AtomicU32::new(512);

/// Physical micro-batch for prefill (`--ubatch`). 512 is llama.cpp's default
/// and what we measured as best on M2 Pro; larger GPUs may prefer 1024.
pub fn set_ubatch(n: Option<u32>) {
    if let Some(n) = n {
        UBATCH.store(n.clamp(32, 4096), Ordering::Relaxed);
    }
}

pub fn ubatch_size() -> u32 {
    UBATCH.load(Ordering::Relaxed)
}

pub struct ModelEngine {
    // Persistent context for prefix caching across generations. Declared
    // before `model`: Rust drops fields in declaration order and the context
    // borrows the model (see the transmute in `stream_generate_with_config`).
    context: Mutex<Option<LlamaContext<'static>>>,
    cached_tokens: Mutex<PrefixCache>,
    backend: Arc<SharedBackend>,
    pub model: Arc<LlamaModel>,
    pub model_path: PathBuf,
    chat: ChatRenderer,
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
        let plan = MemoryPlan::for_model(model_path, use_mlock);
        plan.report();
        let use_mlock = plan.use_mlock;

        let model = load_model_fitted(&backend, model_path, n_gpu_layers, use_mlock, n_ctx)?;
        let model = Arc::new(model);

        // Resolve Auto KV-cache mode
        let resolved_kv = if kv_mode == KvQuantMode::Auto {
            Self::resolve_auto_kv(&model)
        } else {
            kv_mode
        };

        let chat = ChatRenderer::detect(&model);

        Ok(Self {
            context: Mutex::new(None),
            cached_tokens: Mutex::new(PrefixCache::default()),
            backend,
            model,
            model_path: model_path.to_path_buf(),
            chat,
            kv_mode: resolved_kv,
            use_mlock,
            n_gpu_layers,
            n_ctx: AtomicU32::new(n_ctx),
        })
    }

    pub fn chat_format_label(&self) -> &'static str {
        self.chat.label()
    }

    /// Create the persistent context on first use.
    fn ensure_context(&self, slot: &mut Option<LlamaContext<'static>>) -> Result<()> {
        if slot.is_some() {
            return Ok(());
        }
        let n_ctx_val = self.n_ctx.load(Ordering::Relaxed);
        let n_batch = 2048.min(n_ctx_val);
        let n_ubatch = ubatch_size().min(n_batch);
        let p_cores = SiliconProfile::detect().p_cores.max(1) as i32;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(n_ctx_val).unwrap()))
            .with_n_threads(p_cores)
            .with_n_threads_batch(p_cores)
            .with_n_batch(n_batch)
            .with_n_ubatch(n_ubatch)
            .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED)
            // Timing counters cost a little per decode and we measure ourselves
            .with_no_perf(true)
            .with_type_k(self.kv_mode.to_llama_type())
            .with_type_v(self.kv_mode.to_llama_type());
        let ctx = self.model.new_context(&self.backend, ctx_params)?;
        // Safety: model is owned in Arc<LlamaModel> on self, declared after
        // `context`, so it is dropped later.
        let static_ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };
        *slot = Some(static_ctx);
        Ok(())
    }

    /// Where this model's prefix KV state is persisted (`--persist-kv`).
    pub fn session_path(&self) -> Option<PathBuf> {
        let stem = self.model_path.file_stem()?.to_string_lossy().to_string();
        let dir = dirs::home_dir()?.join(".nirvana").join("kv");
        Some(dir.join(format!("{stem}-{}-{:?}.bin", self.n_ctx.load(Ordering::Relaxed), self.kv_mode)))
    }

    /// Persist the first `SESSION_MAX_TOKENS` tokens of the KV state so the
    /// next process starts with the system prompt already evaluated.
    pub fn save_session(&self) -> Result<usize> {
        let Some(path) = self.session_path() else { return Ok(0) };
        let mut ctx_guard = self.context.lock().unwrap();
        let Some(ctx) = ctx_guard.as_mut() else { return Ok(0) };
        let mut cache = self.cached_tokens.lock().unwrap();
        if cache.tokens.is_empty() {
            return Ok(0);
        }
        let keep = cache.tokens.len().min(SESSION_MAX_TOKENS);
        if !cache.rollback(ctx, keep) {
            return Ok(0);
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        ctx.state_seq_save_file(&path, 0, &cache.tokens)
            .map_err(|e| anyhow::anyhow!("save KV state: {e}"))?;
        Ok(keep)
    }

    /// Restore a persisted prefix; returns how many tokens are now warm.
    pub fn load_session(&self) -> Result<usize> {
        let Some(path) = self.session_path() else { return Ok(0) };
        if !path.exists() {
            return Ok(0);
        }
        let mut ctx_guard = self.context.lock().unwrap();
        self.ensure_context(&mut ctx_guard)?;
        let ctx = ctx_guard.as_mut().unwrap();
        let n_ctx = self.n_ctx.load(Ordering::Relaxed) as usize;
        match ctx.state_seq_load_file(&path, 0, n_ctx) {
            Ok((tokens, _)) => {
                let n = tokens.len();
                self.cached_tokens.lock().unwrap().tokens = tokens;
                Ok(n)
            }
            Err(_) => {
                // Stale or incompatible file (different build/quant): drop it
                let _ = std::fs::remove_file(&path);
                ctx.clear_kv_cache();
                Ok(0)
            }
        }
    }

    /// Render a conversation with the model's own chat template.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn format_chat(&self, messages: &[ChatMessage]) -> String {
        self.chat.render(&self.model, messages)
    }

    pub fn stream_chat(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        cancel_token: Arc<AtomicBool>,
        tx: UnboundedSender<StreamEvent>,
    ) -> Result<()> {
        let n_ctx = self.n_ctx.load(Ordering::Relaxed) as usize;
        let prompt = self.chat.render_fitting(&self.model, messages, n_ctx, config.max_tokens);
        self.stream_generate_with_config(&prompt, config, cancel_token, tx)
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
            ..GenerationConfig::default()
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

        // 2. Lock and acquire or initialize persistent context
        let mut ctx_guard = self.context.lock().unwrap();
        self.ensure_context(&mut ctx_guard)?;
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
        let mut sampler = build_sampler(&self.model, config);
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
        // Code repeats itself: a 3-gram hit usually continues for a while, so
        // draft generously; fall back to 2-grams with a short draft.
        const NGRAM_LEN: usize = 3;
        const NGRAM_DRAFT: usize = 8;
        const NGRAM_LEN_FALLBACK: usize = 2;
        const NGRAM_DRAFT_FALLBACK: usize = 4;
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
                ngram_draft(&cache.tokens, current_token, NGRAM_LEN, NGRAM_DRAFT).or_else(|| {
                    ngram_draft(&cache.tokens, current_token, NGRAM_LEN_FALLBACK, NGRAM_DRAFT_FALLBACK)
                })
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
                if !cache.rollback(ctx, keep) {
                    bail!("KV cache refused to evict rejected draft tokens");
                }
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
    Speculative(Arc<SpeculativeEngine>),
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
            let mlx = MlxEngine::load(model_path, ctx_size)?;
            Ok(InferenceEngine::Mlx(Arc::new(mlx)))
        } else {
            let gguf = ModelEngine::load(model_path, gpu_layers, use_mlock, kv_mode, ctx_size)?;
            Ok(InferenceEngine::Gguf(Arc::new(gguf)))
        }
    }

    /// GGUF target with a GGUF draft model for speculative decoding.
    pub fn load_speculative(
        model_path: &Path,
        draft_path: &Path,
        gpu_layers: u32,
        use_mlock: bool,
        kv_mode: KvQuantMode,
        ctx_size: u32,
        n_draft: usize,
    ) -> Result<Self> {
        let spec = SpeculativeEngine::load(model_path, draft_path, gpu_layers, use_mlock, kv_mode, ctx_size, n_draft)?;
        Ok(InferenceEngine::Speculative(Arc::new(spec)))
    }

    /// Raw pre-rendered prompt. Callers that have chat turns should use
    /// [`Self::stream_chat`] so the model's own template is applied.
    #[allow(dead_code)]
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
            InferenceEngine::Speculative(e) => e.stream_generate(prompt, config, cancel_token, tx),
        }
    }

    /// Generate from a conversation, letting each backend render it its own way:
    /// GGUF through the model's chat template, MLX as native chat messages.
    pub fn stream_chat(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        cancel_token: Arc<AtomicBool>,
        tx: UnboundedSender<StreamEvent>,
    ) -> Result<()> {
        match self {
            InferenceEngine::Gguf(e) => e.stream_chat(messages, config, cancel_token, tx),
            InferenceEngine::Mlx(e) => e.stream_chat(messages, config, cancel_token, tx),
            InferenceEngine::Speculative(e) => e.stream_chat(messages, config, cancel_token, tx),
        }
    }

    /// `--persist-kv`: restore the saved prefix state, if any.
    pub fn load_session(&self) -> Result<usize> {
        match self {
            InferenceEngine::Gguf(e) => e.load_session(),
            _ => Ok(0),
        }
    }

    pub fn save_session(&self) -> Result<usize> {
        match self {
            InferenceEngine::Gguf(e) => e.save_session(),
            _ => Ok(0),
        }
    }

    pub fn chat_format_label(&self) -> &'static str {
        match self {
            InferenceEngine::Gguf(e) => e.chat_format_label(),
            InferenceEngine::Mlx(_) => "mlx_lm chat template",
            InferenceEngine::Speculative(e) => e.chat_format_label(),
        }
    }

    pub fn clear_cache(&self) {
        match self {
            InferenceEngine::Gguf(e) => e.clear_cache(),
            InferenceEngine::Mlx(e) => e.clear_cache(),
            InferenceEngine::Speculative(e) => e.clear_cache(),
        }
    }

    pub fn backend_name(&self) -> &'static str {
        match self {
            InferenceEngine::Gguf(_) => "Metal GGUF",
            InferenceEngine::Mlx(_) => "Apple MLX (experimental, via mlx_lm)",
            InferenceEngine::Speculative(_) => "Metal GGUF + draft (speculative)",
        }
    }

    pub fn kv_label(&self) -> String {
        match self {
            InferenceEngine::Gguf(e) => e.kv_mode.label().to_string(),
            InferenceEngine::Mlx(_) => "MLX (managed by mlx_lm)".to_string(),
            InferenceEngine::Speculative(e) => e.kv_mode.label().to_string(),
        }
    }

    #[allow(dead_code)]
    pub fn is_mlx(&self) -> bool {
        matches!(self, InferenceEngine::Mlx(_))
    }

    pub fn total_layers(&self) -> u32 {
        match self {
            InferenceEngine::Gguf(e) => e.total_layers(),
            InferenceEngine::Mlx(e) => e.n_layers,
            InferenceEngine::Speculative(e) => e.target_model.n_layer(),
        }
    }

    pub fn offloaded_layers(&self) -> u32 {
        match self {
            InferenceEngine::Gguf(e) => e.offloaded_layers(),
            InferenceEngine::Mlx(e) => e.n_layers,
            InferenceEngine::Speculative(e) => e.target_model.n_layer().min(e.n_gpu_layers),
        }
    }

    pub fn n_gpu_layers(&self) -> u32 {
        match self {
            InferenceEngine::Gguf(e) => e.n_gpu_layers,
            InferenceEngine::Mlx(e) => e.n_layers,
            InferenceEngine::Speculative(e) => e.n_gpu_layers,
        }
    }

    pub fn use_mlock(&self) -> bool {
        match self {
            InferenceEngine::Gguf(e) => e.use_mlock,
            InferenceEngine::Mlx(_) => false,
            InferenceEngine::Speculative(e) => e.use_mlock,
        }
    }

    pub fn kv_mode(&self) -> KvQuantMode {
        match self {
            InferenceEngine::Gguf(e) => e.kv_mode,
            InferenceEngine::Mlx(_) => KvQuantMode::Auto,
            InferenceEngine::Speculative(e) => e.kv_mode,
        }
    }

    pub fn reconfigure_context(&self, new_n_ctx: u32) -> Result<()> {
        match self {
            InferenceEngine::Gguf(e) => e.reconfigure_context(new_n_ctx),
            InferenceEngine::Mlx(_) => Ok(()),
            InferenceEngine::Speculative(e) => e.reconfigure_context(new_n_ctx),
        }
    }

    pub fn n_ctx(&self) -> u32 {
        match self {
            InferenceEngine::Gguf(e) => e.n_ctx.load(Ordering::Relaxed),
            InferenceEngine::Mlx(e) => e.n_ctx,
            InferenceEngine::Speculative(e) => e.n_ctx.load(Ordering::Relaxed),
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
