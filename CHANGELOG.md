# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `RELEASE_PLAN.md` — codebase review and public release plan.
- MIT license, contributing guide, and GitHub Actions CI (build, clippy, test, catalog URL check).
- `.cargo/config.toml` tracked in the repo.

### Changed
- CLI `--version` now reads from `Cargo.toml` instead of a hardcoded string.
- Clippy warnings resolved across the crate (`format!` inlining, `vec!` literals, argument-count allows).

### Fixed
- **Draft-model speculative decoding produced wrong output.** Verification compared
  the target's logits at index `i` (which predict the token *after* `draft[i]`)
  against `draft[i]` itself, left rejected tokens in the target KV, and fed the
  draft the wrong position. Rewritten on llama.cpp's pending-token model; output
  is now byte-identical to plain greedy decoding (`engine_tests`).
- Speculative contexts are now persistent with prefix caching for both target and
  draft, instead of being rebuilt (cold prefill) on every request.
- Draft/target vocabulary compatibility is checked at load; an incompatible pair
  is refused with a clear error instead of silently generating garbage.
- Prompt-lookup (n-gram) decoding dropped a token whenever every drafted token
  verified, desynchronising the prefix cache from the KV.
- Regenerating an identical prompt sampled from stale logits (full prefix hit
  skipped decoding); at least one prompt token is now always re-evaluated.
- Ending a generation on `max_tokens` or cancel left one un-decoded token in the
  prefix-cache mirror, corrupting the next turn's prefix reuse.
- `ModelEngine` dropped its `LlamaContext` after the model it borrows.
- Fixed sampler seed (`42`) made every regenerate identical; the seed is now random
  per generation, or `--seed N` / `"seed"` in the API for reproducibility.
- Explicit `sampler.accept()` calls removed — `llama_sampler_sample` already accepts,
  so tokens were counted twice.
- Engine errors inside `spawn_blocking` were discarded; they now surface as
  `StreamEvent::Error` in the TUI, CLI, and server.
- Speculative-mode stats reported fabricated prompt/context/mlock values.
- `SiliconProfile::detect()` spawned two `sysctl` processes per generation; it is
  now probed once per process.

### Changed
- Sampler chain reordered to llama.cpp's default (top-k → top-p → min-p → temperature).
- Model-gated engine tests added (`NIRVANA_TEST_MODEL`, `NIRVANA_TEST_DRAFT`).

## [0.2.0]

### Added
- Apple MLX backend via `mlx_lm server` subprocess, with LM Studio model auto-discovery.
- Native attachment pipeline (PDF, image OCR, documents, code) for TUI and Web UI.
- Runtime context reconfiguration (`/v1/context`), project browsing routes, web UI projects sidebar.
- Interactive `/model` selector.

### Fixed
- `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` by aligning batch chunking with context capacity.
- Axum payload limit raised to 100 MB; base64 attachment sanitisation.
