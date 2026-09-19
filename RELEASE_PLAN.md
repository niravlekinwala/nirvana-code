# Nirvana Code — Codebase Review & Public Release Plan

*Review date: 2026-09-19 · Reviewed at commit `b432954` plus uncommitted working tree · Reviewer: Claude (Opus 5)*

---

## 1. Where the project stands

The core idea is sound: llama.cpp on Metal, a persistent context with prefix rollback, Flash Attention on, quantized KV cache. It compiles clean, 9 unit tests pass, and `cargo clippy` reports 78 warnings (all cosmetic). But there are several correctness bugs in the fast paths, the hardware layer is guesswork, and the HTTP server would be a security incident the day it is public.

Also: the README's "bypasses Python wrappers" claim is contradicted by the MLX path, which shells out to `python -m mlx_lm server` over HTTP.

| Area | State |
|---|---|
| Build | `cargo check` clean; 78 clippy warnings (70 are `format!` inlining) |
| Tests | 9 unit tests (attachment, MLX prompt parsing, model detection). No engine tests. |
| Unsafe | 2 `unsafe impl Send/Sync`, 1 `transmute` to `LlamaContext<'static>`, 1 FFI QoS call |
| Repo | No LICENSE, no CI, no CHANGELOG; ~2k lines of uncommitted work; `.cargo/` untracked |
| Dependencies | `llama-cpp-2` 0.1.156 (latest on crates.io as of review) |

---

## 2. Bugs that must be fixed before release

| # | Where | Problem | Impact |
|---|---|---|---|
| 1 | `src/speculative.rs:225` | Off-by-one in verification: logits at index `i` predict the token *after* `candidates[i]`, but are compared against `candidates[i]`. On mismatch the rejected candidate is left in the target KV and the draft is fed the target token at the same position. | Draft-model speculation produces wrong output and near-zero acceptance. Effectively broken. |
| 2 | `src/speculative.rs:84-85` | Both contexts are rebuilt per request — no prefix cache in speculative mode. | Every turn is a cold prefill. |
| 3 | `src/engine.rs:476` | Prompt-lookup: when all N-gram candidates verify, `next_token` is never emitted or pushed to `cached_tokens`, but it *is* in the KV. | Dropped token + prefix-cache/KV desync → later rollbacks land on wrong positions. |
| 4 | `src/engine.rs:342` | Full prefix hit (`tokens_to_eval` empty, e.g. regenerate) skips `decode` entirely; sampling reads stale logits. | Wrong first token on regenerate. llama.cpp convention: always re-evaluate at least the last token. |
| 5 | `src/engine.rs:376`, `src/speculative.rs:154` | `LlamaSampler::dist(42)` — fixed seed. | Every regenerate is byte-identical. Needs `--seed` with random default. |
| 6 | `src/engine.rs:119-121` | `context` is declared after `model`; Rust drops fields in declaration order, so the transmuted `LlamaContext<'static>` outlives the model it references. | Latent use-after-free on engine drop / model switch. Reorder fields (context first). |
| 7 | `src/engine.rs:250`, `src/app.rs:269-288`, `src/server.rs:847` | ChatML hardcoded for every model; `AddBos::Always` regardless of vocab. | Llama-3 / Gemma / Mistral / DeepSeek GGUFs get the wrong template; Qwen gets a spurious BOS. Use the GGUF's own chat template (`LlamaModel::chat_template` + `apply_chat_template`). |
| 8 | `src/hardware.rs:44-58`, `src/engine.rs:292` | Chip identified by substring on the brand string; base M1/M2/M3, M4 Pro/Max, M5 all fall to wrong defaults. Spawns `sysctl` processes on every generation. | Wrong thread counts; wasted latency per request. |
| 9 | `src/speculative.rs:312`, `src/mlx_engine.rs:440` | Stats report fabricated `prompt_tokens: 256/512`, `prefix_cache_hit: true`, `mlock_active: true`. | The UI lies about what happened. |

Smaller items found alongside:

- `src/engine.rs:476` — EOG check uses `current_token` where it should use `next_token`.
- `src/engine.rs` prompt truncation (head 256 + tail) can cut in the middle of a chat turn and corrupt the template.
- `src/engine.rs:110` — `void_logs()` discards all llama.cpp diagnostics; there is no `--verbose`.
- `src/engine.rs` — `InferenceEngine::total_layers()` / `n_ctx()` return hardcoded `32` / `32768` for MLX.
- `src/cli.rs:8` — version string hardcoded `"0.2.0"`; use `env!("CARGO_PKG_VERSION")`.

---

## 3. Security blockers (public release)

1. **Arbitrary file read.** `src/server.rs:611` (`/v1/project/scan`) and `src/server.rs:760` (`/v1/project/file`) accept a client-supplied absolute root. The traversal check only proves the file is under a root *the attacker chose*, so any readable file on disk is served.
2. **Permissive CORS.** `src/server.rs:276` — `CorsLayer::permissive()`. Combined with (1), any web page open in the user's browser can read `~/.ssh/id_rsa` from `localhost:8080`. This is the single most important fix.
3. `/v1/models/load` accepts arbitrary paths; `/v1/chat/completions` will auto-load any model by name (`src/server.rs:915-960`).
4. Unix socket is created with default permissions; the single global `active_cancel` slot means `/v1/chat/stop` aborts whichever request registered last.
5. `src/attachment.rs:303, 416` — `swift -e <script>` per attachment: requires Xcode CLT, JIT-compiles Swift on every PDF/image (seconds), and will not work on a clean user Mac.

---

## 4. The plan — status as of 2026-09-19

All five phases have been executed on the `release-prep` branch. Each phase is
one commit; `git log --oneline` on the branch is the audit trail. What is
**not** done is listed explicitly at the end of this section.

### Phase 0 — Hygiene — ✅ `50090e1`
LICENSE (MIT), CHANGELOG, CONTRIBUTING, CI (clippy `-D warnings`, tests,
portable release build, catalog URL check), version from `Cargo.toml`, zero
clippy warnings, `.cargo/config.toml` tracked with a portability note.

### Phase 1 — Correctness — ✅ `5f0974f`, `5de0411`
All nine bugs fixed. Highlights: speculative decoding rewritten on the
pending-token model and proven byte-identical to plain greedy with a real
0.5B→1.5B draft pair; shared `PrefixCache` (fixed three further latent bugs
found on the way); chat templates rendered from GGUF metadata via minijinja;
hardware probe via `sysctlbyname` + IOKit; honest stats; `--verbose`;
engine errors surfaced instead of swallowed. Model-gated engine tests cover
regenerate, multi-turn prefix reuse, prompt-lookup and speculative equality,
and template detection (verified on Qwen and Gemma 4).

### Phase 2 — Security — ✅ `f234536`
Every item in §3 closed and verified live with curl: workspace confinement
(escape and `..` traversal → 403), CORS off by default, `--api-key` (401
without), `Host` guard (403), installed-only model loads, per-request stop by
id, socket `0600`, Swift helper compiled at build time instead of per call.

### Phase 3 — Performance — ✅ `23a7e6a` (with two items blocked)
| # | Item | Result |
|---|---|---|
| 1 | Working speculative decoding, adaptive K | Done. Correct on real pairs; speed-up expects 7B+ targets (see README). |
| 2 | Prompt-lookup tuning | Done (3-gram/8, fallback 2-gram/4). |
| 3 | Memory fit on 16 GB | Done: `fit_params`, mlock auto-off >70 % RAM, `iogpu.wired_limit_mb` advisory. |
| 4 | Context params | `no_perf` set. `defrag_thold` is deprecated upstream; `swa_full=false` would break arbitrary KV rollback — both left at llama.cpp defaults. `rollback` now honours a refused `seq_rm`. |
| 5 | Prioritised ggml threadpool | **Blocked**: needs the raw `llama_context` pointer, which `llama-cpp-2` keeps `pub(crate)`. Open an upstream PR for an accessor; expected gain was 1–3 %. |
| 6 | KV state persistence | Done: `--persist-kv`, 87 → 22 ms first-turn TTFT in a fresh process. |
| 7 | Batch sizing per chip | `--ubatch` added. Measured on M2 Pro: 512 ≈ 1024 (within noise), 256 slower. Default stays 512; needs M3 Max / M4 numbers from other machines. |
| 8 | Build flags | CI and `scripts/build-release.sh` use `target-cpu=apple-m1`. |
| 9 | Track upstream | dependabot (cargo + actions, weekly). |
| 10 | Sampler chain | Reordered to llama.cpp default; penalties and DRY added, off by default. |

### Phase 4 — MLX — ✅ option (a)
Labelled experimental in the engine, UI strings, and README. `mlx-rs`
evaluation remains a post-release task.

### Phase 5 — Release readiness — ✅ `d10f098`, `7b938ef`, this commit
Reproducible `bench --runs --json`; all 13 catalog URLs verified; downloads
resume and verify SHA-256 against Hugging Face's `X-Linked-ETag`; config file;
web UI audited (one real XSS fixed) and split into three files; Homebrew
formula template; release build script; README rewritten around measured
numbers and the security model.

### Still open (needs the maintainer, not code)
1. **Publish**: create the GitHub repo/tap, replace `<you>` in `README.md` and
   `packaging/nirvana-code.rb`, run `scripts/build-release.sh --sign`, notarize,
   fill in the formula's URL and sha256.
2. **Cross-chip numbers**: run `nirvana-code bench --runs 5 --json` on M1, M3
   Max, M4 and paste into the README table; revisit the `--ubatch` default.
3. **Speculative speed-up on a large target**: download Qwen2.5-Coder-7B (same
   tokenizer as the 0.5B draft) and record the number before promoting the
   feature.
4. **Threadpool priority**: upstream accessor in `llama-cpp-2` (Phase 3 §5).
5. **Prompt templates**: `templates.rs` targets Claude 3.7 / o1 / Antigravity —
   dated names; refresh or drop.
6. **Merge** `release-prep` into `main` and tag `v0.3.0`.

## 5. Timeline

Planned 3–4 weeks; executed in one session on 2026-09-19 across seven commits.
The remaining items above are publishing and measurement tasks, not code.
