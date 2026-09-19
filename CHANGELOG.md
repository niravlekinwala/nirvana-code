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
- _(Phase 1 — see `RELEASE_PLAN.md`)_

## [0.2.0]

### Added
- Apple MLX backend via `mlx_lm server` subprocess, with LM Studio model auto-discovery.
- Native attachment pipeline (PDF, image OCR, documents, code) for TUI and Web UI.
- Runtime context reconfiguration (`/v1/context`), project browsing routes, web UI projects sidebar.
- Interactive `/model` selector.

### Fixed
- `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` by aligning batch chunking with context capacity.
- Axum payload limit raised to 100 MB; base64 attachment sanitisation.
