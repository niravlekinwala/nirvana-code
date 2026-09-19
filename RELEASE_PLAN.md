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

## 4. The plan

### Phase 0 — Hygiene (≈1 day) — ✅ done (`50090e1`)

- Commit the ~2k lines of uncommitted work on a branch. Add `.cargo/config.toml` to the repo deliberately (see Phase 3, item 8, on `target-cpu=native`).
- `cargo clippy --fix`; `src/cli.rs:8` version → `env!("CARGO_PKG_VERSION")`.
- Add LICENSE, CHANGELOG, CONTRIBUTING.
- GitHub Actions on a `macos-14` (arm64) runner: build, `clippy -D warnings`, test, and a HEAD-check of every `MODEL_CATALOG` URL.

### Phase 1 — Correctness (≈3–5 days) — 🟡 in progress

Done: bugs 1, 2, 3, 4, 5, 6, 9 (speculative rewrite, shared `PrefixCache`,
seed, field order, honest stats), plus the engine tests. Remaining: 7 (chat
templates), 8 (hardware probe rewrite — detection is now memoised but still
string-matched), prompt truncation on turn boundaries, `--verbose` logging.

Measured on M2 Pro / Qwen2.5-Coder-1.5B / 0.5B draft, temp 0: plain 116 tok/s,
speculative K=4 81 tok/s at 66 % acceptance. A 1.5B target is too fast for a
draft to pay off; the win is expected on 7B–27B targets (Phase 3 §1).

Fix bugs 1–9 above. Concretely:

- **Rewrite `speculative.rs` verification**: sample the target from the *previous* step's logits and compare with `cand[0]`; then logits index `i` vs `cand[i+1]`. On rejection, `seq_rm` from `pos + accepted` on both contexts and feed the target token to the draft. Make `SpeculativeEngine` own persistent target/draft contexts with the same prefix-cache logic as `ModelEngine`.
- **Factor a shared `PrefixCache` struct** (token list + longest-common-prefix + rollback). The logic is currently duplicated across `engine.rs`, `speculative.rs`, and the context-shift path.
- **Full-prefix-hit**: if `common_prefix_len == n_prompt`, decrement by one and re-decode the last token.
- **Prompt-lookup**: emit and cache `next_token` in the all-verified branch; fix the EOG check.
- **Chat templates**: build prompts via the GGUF's template; read `add_bos` from the vocab.
- **Seed**: `--seed` flag, random by default; plumb through `GenerationConfig`.
- **Hardware**: replace `hardware.rs` with `libc::sysctlbyname` reads of `hw.perflevel0.physicalcpu` (P-cores), `hw.perflevel1.physicalcpu` (E-cores), `hw.memsize`, and Metal's `recommendedMaxWorkingSetSize` via `objc2-metal`. Cache the result in a `OnceLock`.
- **Field order** in `ModelEngine`: `context` before `model`.
- **Engine tests** using the 0.5B Qwen (gated behind `NIRVANA_TEST_MODEL`): prefix hit / miss / regenerate; ngram all-verified path; speculative output equals greedy non-speculative output at `temperature = 0`.

### Phase 2 — Security & server (≈2–3 days)

- `--workspace <dir>` at startup; project endpoints are confined to it. Canonicalize then `starts_with` to reject symlink escapes.
- CORS: default to same-origin only; `--cors-origin <origin>` to opt in.
- `--api-key` bearer token; auto-generate and print one when bound to a non-loopback host. Validate the `Host` header against DNS rebinding.
- Restrict `/v1/models/load` and auto-load to the model directories `ModelManager` already knows.
- Per-request cancel tokens keyed by request id; `chmod 0600` on the Unix socket.
- Replace `swift -e` with in-process `objc2-pdf-kit` / `objc2-vision` (same Apple frameworks, no compiler, ~100× faster), or ship a prebuilt helper binary.

### Phase 3 — Silicon-level performance

Ordered by expected payoff. The honest framing: llama.cpp's Metal kernels are already the fast path — the leverage is in *how the engine drives them*, not in writing shaders.

| # | Change | Expected effect |
|---|---|---|
| 1 | **Working speculative decoding** (after the Phase 1 fix). Adaptive `n_draft`: grow on accept, shrink on reject, cap 16. | 1.5–2.5× decode on 7B–27B targets with the 0.5B draft; higher on code. |
| 2 | **Prompt-lookup tuning**: ngram 3 with fallback to 2, `draft_len` 8–12, hashmap index instead of the O(n) backward scan in `engine.rs`. | 1.3–2× on edit/refactor tasks at long context. |
| 3 | **Memory fit on 16 GB**: use `LlamaModelParams::fit_params` (present in the crate) to auto-size `n_ctx`/offload; auto-disable `mlock` when model > ~70 % of RAM; detect and *advise* `sudo sysctl iogpu.wired_limit_mb=…` since macOS caps GPU-wired memory at ~75 % (≈67 % on ≤36 GB machines). | The difference between "27B runs" and "27B swaps". |
| 4 | **Context params not yet set**: `.with_no_perf(true)`, `.with_swa_full(false)` for SWA models, `.with_op_offload(true)`, `.with_defrag_thold(0.1)` so long sessions don't degrade after many `seq_rm`s. | Small steady-state wins; prevents slow decay. |
| 5 | **Threads**: the QoS call at `engine.rs:286` only affects the calling thread; ggml's workers are separate. Use `llama_attach_threadpool` with a P-core cpumask and `GGML_SCHED_PRIO_HIGH`. | 1–3 %; removes E-core jitter in TTFT. |
| 6 | **KV state persistence**: `state_seq_save_file` for the system-prompt prefix so a cold start after relaunch is a warm hit. | Makes the 388 → 42 ms number the *default* first-turn experience. |
| 7 | **Batch sizing per chip**: benchmark `n_ubatch` 512 vs 1024 on M2 Pro / M3 Max / M4; key a profile table on GPU-core count. | Prefill throughput on larger GPUs. |
| 8 | **Build flags**: `-C target-cpu=native` in `.cargo/config.toml` makes a release binary built on M4 (SME) crash on M1. Distribute with `target-cpu=apple-m1` and let ggml runtime-dispatch; keep `native` for `cargo install`. Verify `GGML_METAL_EMBED_LIBRARY` so the binary carries its metallib. | Portable binaries. |
| 9 | **Track upstream**: bump `llama-cpp-2` on a schedule; Metal FA and MoE kernel improvements (relevant to the Ornith / LFM catalog entries) arrive that way. | Free wins. |
| 10 | **Sampler chain**: reorder to llama.cpp's default (top-k → top-p → min-p → temp); add optional `penalties` / `dry` (both in the crate) instead of the MLX-only `repetition_penalty`. | Output quality / parity across backends. |

### Phase 4 — MLX decision

Today the MLX backend is a Python subprocess with a 25 s startup and no in-process control. Options:

- **(a)** Mark it experimental and label it honestly in README and UI — *do this now*.
- **(b)** Move to `mlx-rs` for in-process MLX — evaluate after Phase 3.
- **(c)** Drop it for 1.0.

Don't let it block release.

### Phase 5 — Release readiness (≈1 week)

- **Reproducible benchmark**: `nirvana-code bench --json --runs 5`; report median prefill tok/s, decode tok/s, and TTFT cold/warm separately. Publish a table for M1 / M2 Pro / M3 Max / M4 so README claims are verifiable.
- **Catalog**: verify every `MODEL_CATALOG` URL exists (several entries — "Qwen 3.8 27B", "Ornith 1.5", the "AtomicChat" org — could not be confirmed during review). Add SHA256 checksums and resume support to `download_file`.
- **Config file**: `~/.config/nirvana-code/config.toml`; `--verbose` routing llama.cpp logs through `tracing` instead of `void_logs()`.
- **Web UI**: split `src/web/index.html` (3010 lines) into files; audit the 21 `innerHTML` sites — `escapeHtml` exists, confirm every model-output path goes through it.
- **Distribution**: Homebrew tap + `cargo install`; codesign and notarize the binary.
- **README**: reword architecture claims to match what the code does; document the security model of `serve`.

---

## 5. Timeline

| Phase | Effort | Gate |
|---|---|---|
| 0 Hygiene | 1 day | — |
| 1 Correctness | 3–5 days | **Required for release** |
| 2 Security | 2–3 days | **Required for release** |
| 3 Performance | 1–2 weeks | Items 1, 3, 8 before release; rest can follow |
| 4 MLX | 0.5 day (option a) | — |
| 5 Release readiness | 1 week | Required for release |

Rough total: 3–4 weeks of focused work. Phases 1–2 are the non-negotiable gate.

Recommended starting point: the speculative-decoding rewrite and the prefix-cache fixes (Phase 1) — highest value, and they unblock Phase 3 item 1.
