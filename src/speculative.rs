use anyhow::{bail, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc::UnboundedSender;

use crate::chat::{ChatMessage, ChatRenderer};
use crate::engine::{
    boost_thread_qos, build_sampler, load_model_fitted, GenerationConfig, KvQuantMode, MemoryPlan,
    ModelEngine, PrefixCache, SharedBackend, StreamEvent,
};
use crate::hardware::SiliconProfile;

/// llama.cpp tolerates small vocab-size differences between draft and target
/// (padding rows); anything larger means the models do not share a tokenizer.
const MAX_VOCAB_SIZE_DIFF: i32 = 128;

/// Adaptive draft length bounds. Grows by one after a round where every draft
/// token was accepted, shrinks by one after an early rejection.
const MIN_DRAFT: usize = 1;
const MAX_DRAFT: usize = 16;

/// Persistent context plus the mirror of what is in its KV cache.
struct ModelSlot {
    ctx: Option<LlamaContext<'static>>,
    cache: PrefixCache,
}

/// Two-model speculative decoding: a small draft model proposes `n_draft`
/// tokens greedily, the target verifies them in one batched forward pass.
///
/// Both contexts persist across calls so multi-turn chats reuse the prefix
/// exactly like `ModelEngine`.
pub struct SpeculativeEngine {
    // Declared before the models: contexts borrow them (see the transmute in
    // `ensure_context`) and Rust drops fields in declaration order.
    target: Mutex<ModelSlot>,
    draft: Mutex<ModelSlot>,
    backend: Arc<SharedBackend>,
    pub target_model: Arc<LlamaModel>,
    pub draft_model: Arc<LlamaModel>,
    chat: ChatRenderer,
    pub n_gpu_layers: u32,
    pub use_mlock: bool,
    pub kv_mode: KvQuantMode,
    pub n_ctx: AtomicU32,
    /// Current draft length; adapts between rounds and persists across calls.
    n_draft: AtomicUsize,
}

unsafe impl Send for SpeculativeEngine {}
unsafe impl Sync for SpeculativeEngine {}

impl SpeculativeEngine {
    pub fn load(
        target_path: &Path,
        draft_path: &Path,
        n_gpu_layers: u32,
        use_mlock: bool,
        kv_mode: KvQuantMode,
        n_ctx: u32,
        n_draft: usize,
    ) -> Result<Self> {
        let backend = SharedBackend::get()?;
        let plan = MemoryPlan::for_model(target_path, use_mlock);
        plan.report();
        let use_mlock = plan.use_mlock;

        let target_model = load_model_fitted(&backend, target_path, n_gpu_layers, use_mlock, n_ctx)?;
        // The draft is small; pin it fully alongside the target.
        let draft_params = LlamaModelParams::default()
            .with_n_gpu_layers(n_gpu_layers)
            .with_use_mlock(use_mlock);
        let draft_model = LlamaModel::load_from_file(&backend, draft_path, &draft_params)?;
        Self::check_vocab_compat(&target_model, &draft_model)?;

        let kv_mode = if kv_mode == KvQuantMode::Auto {
            ModelEngine::resolve_auto_kv(&target_model)
        } else {
            kv_mode
        };
        let chat = ChatRenderer::detect(&target_model);

        Ok(Self {
            target: Mutex::new(ModelSlot { ctx: None, cache: PrefixCache::default() }),
            draft: Mutex::new(ModelSlot { ctx: None, cache: PrefixCache::default() }),
            backend,
            target_model: Arc::new(target_model),
            draft_model: Arc::new(draft_model),
            chat,
            n_gpu_layers,
            use_mlock,
            kv_mode,
            n_ctx: AtomicU32::new(n_ctx),
            n_draft: AtomicUsize::new(n_draft.clamp(MIN_DRAFT, MAX_DRAFT)),
        })
    }

    /// The draft's token ids are fed straight into the target, so the two
    /// vocabularies must agree. Mirrors the check in llama.cpp's `common/speculative`.
    fn check_vocab_compat(target: &LlamaModel, draft: &LlamaModel) -> Result<()> {
        let (nt, nd) = (target.n_vocab(), draft.n_vocab());
        if (nt - nd).abs() > MAX_VOCAB_SIZE_DIFF {
            bail!(
                "Draft model vocab ({nd}) does not match target vocab ({nt}); speculative decoding needs models that share a tokenizer"
            );
        }
        if target.token_bos() != draft.token_bos() || target.token_eos() != draft.token_eos() {
            bail!("Draft and target models use different BOS/EOS tokens; they do not share a tokenizer");
        }
        // Spot-check token text across the shared range
        let n = nt.min(nd);
        for id in (5..n).step_by(997) {
            let tok = LlamaToken(id);
            let a = target.token_to_piece_bytes(tok, 64, true, None).ok();
            let b = draft.token_to_piece_bytes(tok, 64, true, None).ok();
            if a != b {
                bail!("Draft and target tokenizers disagree on token {id}; speculative decoding needs a matching vocabulary");
            }
        }
        Ok(())
    }

    /// Explicitly clear both prefix caches
    pub fn clear_cache(&self) {
        for slot in [&self.target, &self.draft] {
            if let Ok(mut s) = slot.lock() {
                if let Some(ctx) = s.ctx.as_mut() {
                    ctx.clear_kv_cache();
                }
                s.cache.tokens.clear();
            }
        }
    }

    /// Drop both contexts so they are rebuilt at the new size on next use.
    pub fn reconfigure_context(&self, new_n_ctx: u32) -> Result<()> {
        for slot in [&self.target, &self.draft] {
            let mut s = slot.lock().unwrap();
            s.ctx = None;
            s.cache.tokens.clear();
        }
        self.n_ctx.store(new_n_ctx, Ordering::Relaxed);
        Ok(())
    }

    pub fn chat_format_label(&self) -> &'static str {
        self.chat.label()
    }

    pub fn stream_chat(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        cancel_token: Arc<AtomicBool>,
        tx: UnboundedSender<StreamEvent>,
    ) -> Result<()> {
        let n_ctx = self.n_ctx.load(Ordering::Relaxed) as usize;
        let prompt = self.chat.render_fitting(&self.target_model, messages, n_ctx, config.max_tokens);
        self.stream_generate(&prompt, config, cancel_token, tx)
    }

    fn context_params(&self, n_batch: u32, n_ubatch: u32) -> LlamaContextParams {
        let p_cores = SiliconProfile::detect().p_cores.max(1) as i32;
        LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(self.n_ctx.load(Ordering::Relaxed)).unwrap()))
            .with_n_threads(p_cores)
            .with_n_threads_batch(p_cores)
            .with_n_batch(n_batch)
            .with_n_ubatch(n_ubatch)
            .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED)
            .with_type_k(self.kv_mode.to_llama_type())
            .with_type_v(self.kv_mode.to_llama_type())
    }

    fn ensure_context(
        &self,
        slot: &mut ModelSlot,
        model: &Arc<LlamaModel>,
        n_batch: u32,
        n_ubatch: u32,
    ) -> Result<()> {
        if slot.ctx.is_none() {
            let ctx = model.new_context(&self.backend, self.context_params(n_batch, n_ubatch))?;
            // Safety: the model is held in an Arc on `self`, which outlives the
            // context because the slot fields are dropped first.
            let static_ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };
            slot.ctx = Some(static_ctx);
        }
        Ok(())
    }

    pub fn stream_generate(
        &self,
        prompt: &str,
        config: &GenerationConfig,
        cancel_token: Arc<AtomicBool>,
        tx: UnboundedSender<StreamEvent>,
    ) -> Result<()> {
        let start_time = Instant::now();
        boost_thread_qos();

        // 1. Tokenize once with the target; the draft shares the vocabulary
        let prompt_tokens = match self.target_model.str_to_token(prompt, AddBos::Always) {
            Ok(tokens) => tokens,
            Err(e) => {
                let _ = tx.send(StreamEvent::Error(format!("Tokenization failed: {e}")));
                bail!("Tokenization failed");
            }
        };
        let n_prompt = prompt_tokens.len();
        if n_prompt == 0 {
            let _ = tx.send(StreamEvent::Done);
            return Ok(());
        }
        let n_ctx_val = self.n_ctx.load(Ordering::Relaxed);
        let n_ctx = n_ctx_val as usize;
        // Room for the pending token plus the largest draft on every round
        let reserve = MAX_DRAFT + 2;
        if n_prompt + reserve >= n_ctx {
            let _ = tx.send(StreamEvent::Error(format!(
                "Prompt ({n_prompt} tokens) does not fit in the {n_ctx}-token context"
            )));
            bail!("Prompt exceeds context");
        }

        let n_batch = 2048.min(n_ctx_val);
        let n_ubatch = 512.min(n_batch);
        let batch_size = (n_batch as usize).min(512);

        // 2. Persistent contexts and prefix caches for both models
        let mut target = self.target.lock().unwrap();
        let mut draft = self.draft.lock().unwrap();
        self.ensure_context(&mut target, &self.target_model, n_batch, n_ubatch)?;
        self.ensure_context(&mut draft, &self.draft_model, n_batch, n_ubatch)?;
        let ModelSlot { ctx: t_ctx, cache: t_cache } = &mut *target;
        let ModelSlot { ctx: d_ctx, cache: d_cache } = &mut *draft;
        let t_ctx = t_ctx.as_mut().unwrap();
        let d_ctx = d_ctx.as_mut().unwrap();

        let mut t_batch = LlamaBatch::new(batch_size.max(MAX_DRAFT + 1), 1);
        let mut d_batch = LlamaBatch::new(batch_size, 1);

        let prefix_tokens_reused = t_cache.sync(t_ctx, &prompt_tokens);
        if !t_cache.prefill(t_ctx, &mut t_batch, &prompt_tokens[prefix_tokens_reused..], batch_size, &cancel_token)? {
            let _ = tx.send(StreamEvent::Done);
            return Ok(());
        }
        let d_reused = d_cache.sync(d_ctx, &prompt_tokens);
        if !d_cache.prefill(d_ctx, &mut d_batch, &prompt_tokens[d_reused..], batch_size, &cancel_token)? {
            let _ = tx.send(StreamEvent::Done);
            return Ok(());
        }

        // 3. Samplers: the draft is always greedy — its job is to guess what the
        //    target will pick, and the target's own sampler makes the real choice.
        let mut target_sampler = build_sampler(config);
        let mut draft_sampler = LlamaSampler::greedy();
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        let mut total_generated = 0usize;
        let mut drafted = 0usize;
        let mut accepted = 0usize;
        let mut n_draft = self.n_draft.load(Ordering::Relaxed);

        let mut emit = |tok: LlamaToken, total: &mut usize| {
            *total += 1;
            if let Ok(piece) = self.target_model.token_to_piece(tok, &mut decoder, false, None) {
                if !piece.is_empty() {
                    let _ = tx.send(StreamEvent::Token(piece));
                }
            }
        };

        // `pending` is the target's latest sample: emitted, not yet in either KV.
        let mut pending = target_sampler.sample(t_ctx, -1);
        if self.target_model.is_eog_token(pending) {
            let _ = tx.send(StreamEvent::Done);
            return Ok(());
        }
        let first_token_time = Instant::now();
        emit(pending, &mut total_generated);

        // 4. Speculative loop
        while total_generated < config.max_tokens {
            if cancel_token.load(Ordering::Relaxed) {
                break;
            }
            let n_past = t_cache.tokens.len();
            if n_past + reserve >= n_ctx {
                // Context is full; no context shifting in speculative mode
                break;
            }

            // a. Bring the draft in line with target KV + pending. After a
            //    rejection this rolls back the mis-drafted tail; in all cases at
            //    least `pending` is decoded so the draft's logits are fresh.
            let mut want = Vec::with_capacity(n_past + 1);
            want.extend_from_slice(&t_cache.tokens);
            want.push(pending);
            let d_common = d_cache.sync(d_ctx, &want);
            if !d_cache.prefill(d_ctx, &mut d_batch, &want[d_common..], batch_size, &cancel_token)? {
                break;
            }

            // b. Draft up to n_draft tokens greedily
            let mut cands: Vec<LlamaToken> = Vec::with_capacity(n_draft);
            while cands.len() < n_draft {
                let d_tok = draft_sampler.sample(d_ctx, -1);
                if self.draft_model.is_eog_token(d_tok) {
                    break;
                }
                cands.push(d_tok);
                if cands.len() == n_draft {
                    break;
                }
                d_batch.clear();
                d_batch.add(d_tok, d_cache.tokens.len() as i32, &[0], true)?;
                d_ctx.decode(&mut d_batch)?;
                d_cache.tokens.push(d_tok);
            }
            drafted += cands.len();

            // c. One target pass over [pending, cands...]. Logits at index i
            //    predict the token after batch[i], i.e. they verify cands[i].
            t_batch.clear();
            t_batch.add(pending, n_past as i32, &[0], true)?;
            for (k, &cand) in cands.iter().enumerate() {
                t_batch.add(cand, (n_past + 1 + k) as i32, &[0], true)?;
            }
            t_ctx.decode(&mut t_batch)?;
            t_cache.tokens.push(pending);

            // d. Verify left to right; the first disagreement (or the sample
            //    after the last accepted draft) becomes the new pending token.
            let mut finished = false;
            let mut next = pending;
            for i in 0..=cands.len() {
                next = target_sampler.sample(t_ctx, i as i32);
                if self.target_model.is_eog_token(next) {
                    finished = true;
                    break;
                }
                if i < cands.len() && next == cands[i] {
                    t_cache.tokens.push(cands[i]);
                    accepted += 1;
                    emit(cands[i], &mut total_generated);
                    if total_generated >= config.max_tokens {
                        finished = true;
                        break;
                    }
                } else {
                    break;
                }
            }

            // Evict rejected draft tokens from the target KV
            let keep = t_cache.tokens.len();
            let n_acc = keep - n_past - 1;
            t_cache.rollback(t_ctx, keep);
            if finished {
                break;
            }

            // Adapt: a clean sweep earns a longer draft, an early miss a shorter one
            if !cands.is_empty() {
                if n_acc == cands.len() {
                    n_draft = (n_draft + 1).min(MAX_DRAFT);
                } else if n_acc + 1 < cands.len() {
                    n_draft = n_draft.saturating_sub(1).max(MIN_DRAFT);
                }
            }

            pending = next;
            emit(pending, &mut total_generated);
        }

        self.n_draft.store(n_draft, Ordering::Relaxed);

        // 5. Stats
        let ttft_ms = first_token_time.duration_since(start_time).as_millis();
        let decode_secs = first_token_time.elapsed().as_secs_f64();
        let tps = if total_generated > 1 && decode_secs > 0.0 {
            (total_generated - 1) as f64 / decode_secs
        } else {
            0.0
        };
        let acceptance = if drafted > 0 {
            accepted as f64 / drafted as f64 * 100.0
        } else {
            0.0
        };

        let _ = tx.send(StreamEvent::Stats {
            ttft_ms,
            tokens_per_sec: tps,
            total_tokens: total_generated,
            prompt_tokens: n_prompt,
            context_used: t_cache.tokens.len(),
            context_capacity: n_ctx_val,
            prefix_tokens_reused,
            prefix_cache_hit: prefix_tokens_reused > 0,
            kv_type: format!(
                "{} · Speculative K={} (acc {:.0}%)",
                self.kv_mode.label(),
                n_draft,
                acceptance
            ),
            mlock_active: self.use_mlock,
        });
        let _ = tx.send(StreamEvent::Done);
        Ok(())
    }
}
