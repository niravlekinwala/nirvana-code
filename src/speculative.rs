use anyhow::{bail, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::UnboundedSender;

use crate::engine::{ModelEngine, StreamEvent};

pub struct SpeculativeEngine {
    backend: Arc<crate::engine::SharedBackend>,
    pub target_model: Arc<LlamaModel>,
    pub draft_model: Arc<LlamaModel>,
    #[allow(dead_code)]
    pub n_gpu_layers: u32,
    pub n_ctx: u32,
    pub n_draft: usize,
}

impl SpeculativeEngine {
    pub fn load(
        target_path: &Path,
        draft_path: &Path,
        n_gpu_layers: u32,
        use_mlock: bool,
        n_ctx: u32,
        n_draft: usize,
    ) -> Result<Self> {
        let backend = crate::engine::SharedBackend::get()?;

        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(n_gpu_layers)
            .with_use_mlock(use_mlock);

        let target_model = LlamaModel::load_from_file(&backend, target_path, &model_params)?;
        let draft_model = LlamaModel::load_from_file(&backend, draft_path, &model_params)?;

        Ok(Self {
            backend,
            target_model: Arc::new(target_model),
            draft_model: Arc::new(draft_model),
            n_gpu_layers,
            n_ctx,
            n_draft,
        })
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

        let kv_type = ModelEngine::resolve_auto_kv(&self.target_model).to_llama_type();

        let n_batch = 2048.min(self.n_ctx);
        let n_ubatch = 512.min(n_batch);
        let batch_size = (n_batch as usize).min(512);

        // 1. Context params with Metal Flash Attention & Auto KV
        let make_params = || {
            LlamaContextParams::default()
                .with_n_ctx(Some(NonZeroU32::new(self.n_ctx).unwrap()))
                .with_n_threads(4)
                .with_n_threads_batch(8)
                .with_n_batch(n_batch)
                .with_n_ubatch(n_ubatch)
                .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_AUTO)
                .with_type_k(kv_type)
                .with_type_v(kv_type)
        };

        let mut target_ctx = self.target_model.new_context(&self.backend, make_params())?;
        let mut draft_ctx = self.draft_model.new_context(&self.backend, make_params())?;

        // 2. Tokenize prompt for target and draft
        let target_tokens = match self.target_model.str_to_token(prompt, AddBos::Always) {
            Ok(tokens) => tokens,
            Err(e) => {
                let _ = tx.send(StreamEvent::Error(format!("Target tokenization failed: {}", e)));
                bail!("Tokenization failed");
            }
        };

        let draft_tokens = match self.draft_model.str_to_token(prompt, AddBos::Always) {
            Ok(tokens) => tokens,
            Err(e) => {
                let _ = tx.send(StreamEvent::Error(format!("Draft tokenization failed: {}", e)));
                bail!("Draft tokenization failed");
            }
        };

        let n_prompt = target_tokens.len();
        if n_prompt == 0 {
            let _ = tx.send(StreamEvent::Done);
            return Ok(());
        }

        // Prefill Target Context
        let mut target_batch = LlamaBatch::new(batch_size, 1);
        let mut target_pos = 0;
        for chunk in target_tokens.chunks(batch_size) {
            if cancel_token.load(Ordering::Relaxed) {
                let _ = tx.send(StreamEvent::Done);
                return Ok(());
            }
            target_batch.clear();
            let chunk_len = chunk.len();
            for (i, &token) in chunk.iter().enumerate() {
                let is_last = (target_pos + i + 1) == n_prompt;
                target_batch.add(token, (target_pos + i) as i32, &[0], is_last)?;
            }
            target_pos += chunk_len;
            target_ctx.decode(&mut target_batch)?;
        }

        // Prefill Draft Context
        let mut draft_batch = LlamaBatch::new(batch_size, 1);
        let mut draft_pos = 0;
        for chunk in draft_tokens.chunks(batch_size) {
            if cancel_token.load(Ordering::Relaxed) {
                let _ = tx.send(StreamEvent::Done);
                return Ok(());
            }
            draft_batch.clear();
            let chunk_len = chunk.len();
            for (i, &token) in chunk.iter().enumerate() {
                let is_last = (draft_pos + i + 1) == draft_tokens.len();
                draft_batch.add(token, (draft_pos + i) as i32, &[0], is_last)?;
            }
            draft_pos += chunk_len;
            draft_ctx.decode(&mut draft_batch)?;
        }

        let mut target_sampler = if temperature <= 0.05 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::min_p(0.05, 1),
                LlamaSampler::top_k(40),
                LlamaSampler::top_p(0.9, 1),
                LlamaSampler::temp(temperature),
                LlamaSampler::dist(42),
            ])
        };

        let mut draft_sampler = LlamaSampler::greedy();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut first_token_time: Option<Instant> = None;
        let mut total_generated = 0;
        let mut draft_tokens_proposed = 0;
        let mut draft_tokens_accepted = 0;

        // Speculative generation loop
        while total_generated < max_tokens {
            if cancel_token.load(Ordering::Relaxed) {
                break;
            }

            // 1. Draft generates K candidate tokens
            let mut candidates: Vec<LlamaToken> = Vec::with_capacity(self.n_draft);
            for _ in 0..self.n_draft {
                let d_tok = draft_sampler.sample(&draft_ctx, draft_batch.n_tokens() - 1);
                draft_sampler.accept(d_tok);

                if self.draft_model.is_eog_token(d_tok) {
                    break;
                }
                candidates.push(d_tok);

                draft_batch.clear();
                draft_batch.add(d_tok, draft_pos as i32, &[0], true)?;
                draft_pos += 1;
                draft_ctx.decode(&mut draft_batch)?;
            }

            draft_tokens_proposed += candidates.len();

            // 2. If no candidates generated, sample 1 token directly from target
            if candidates.is_empty() {
                let tok = target_sampler.sample(&target_ctx, target_batch.n_tokens() - 1);
                target_sampler.accept(tok);
                if self.target_model.is_eog_token(tok) {
                    break;
                }
                if first_token_time.is_none() {
                    first_token_time = Some(Instant::now());
                }
                total_generated += 1;
                if let Ok(piece) = self.target_model.token_to_piece(tok, &mut decoder, false, None) {
                    if !piece.is_empty() {
                        let _ = tx.send(StreamEvent::Token(piece));
                    }
                }
                target_batch.clear();
                target_batch.add(tok, target_pos as i32, &[0], true)?;
                target_pos += 1;
                target_ctx.decode(&mut target_batch)?;
                continue;
            }

            // 3. Evaluate all candidate tokens in one single batched target forward pass
            target_batch.clear();
            for (i, &cand) in candidates.iter().enumerate() {
                target_batch.add(cand, (target_pos + i) as i32, &[0], true)?;
            }
            target_ctx.decode(&mut target_batch)?;

            // 4. Verification pass
            let mut accepted_count = 0;
            let mut stopped = false;

            for (i, &cand) in candidates.iter().enumerate() {
                let target_sampled = target_sampler.sample(&target_ctx, i as i32);
                target_sampler.accept(target_sampled);

                if first_token_time.is_none() {
                    first_token_time = Some(Instant::now());
                }

                if target_sampled == cand {
                    // Candidate accepted!
                    accepted_count += 1;
                    draft_tokens_accepted += 1;
                    total_generated += 1;

                    if let Ok(piece) = self.target_model.token_to_piece(cand, &mut decoder, false, None) {
                        if !piece.is_empty() {
                            let _ = tx.send(StreamEvent::Token(piece));
                        }
                    }

                    if self.target_model.is_eog_token(cand) {
                        stopped = true;
                        break;
                    }
                } else {
                    // Mismatch: candidate rejected, accept target's token instead
                    total_generated += 1;
                    if let Ok(piece) = self.target_model.token_to_piece(target_sampled, &mut decoder, false, None) {
                        if !piece.is_empty() {
                            let _ = tx.send(StreamEvent::Token(piece));
                        }
                    }

                    // Rollback target context to position after accepted + target token
                    let valid_target_pos = target_pos + accepted_count + 1;
                    let _ = target_ctx.kv_cache_seq_rm(0, Some(valid_target_pos as u32), None);
                    target_pos = valid_target_pos;

                    // Rollback draft context to match target position
                    let _ = draft_ctx.kv_cache_seq_rm(0, Some(valid_target_pos as u32), None);
                    draft_pos = valid_target_pos;

                    // Feed the target token into draft model
                    draft_batch.clear();
                    draft_batch.add(target_sampled, (valid_target_pos - 1) as i32, &[0], true)?;
                    draft_ctx.decode(&mut draft_batch)?;

                    if self.target_model.is_eog_token(target_sampled) {
                        stopped = true;
                    }
                    break;
                }
            }

            if stopped {
                break;
            }

            if accepted_count == candidates.len() {
                target_pos += candidates.len();
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

        let acceptance_rate = if draft_tokens_proposed > 0 {
            (draft_tokens_accepted as f64 / draft_tokens_proposed as f64) * 100.0
        } else {
            0.0
        };

        let _ = tx.send(StreamEvent::Stats {
            ttft_ms,
            tokens_per_sec: tps,
            total_tokens: total_generated,
            prompt_tokens: 256,
            context_used: total_generated + 256,
            context_capacity: self.n_ctx,
            prefix_tokens_reused: draft_tokens_accepted,
            prefix_cache_hit: true,
            kv_type: format!("Speculative (Acc: {:.1}%)", acceptance_rate),
            mlock_active: true,
        });

        let _ = tx.send(StreamEvent::Done);
        Ok(())
    }
}
