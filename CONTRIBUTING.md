# Contributing to Nirvana Code

Thanks for your interest. This project targets Apple Silicon only; contributions
are expected to build and run on an M-series Mac.

## Prerequisites

- macOS 14+ on Apple Silicon
- Rust stable (edition 2024 → 1.85+), `cmake`, and Xcode Command Line Tools
- A small GGUF model for local testing: `nirvana-code download qwen-0.5b`

## Workflow

1. Branch from `main`.
2. Keep `cargo clippy --all-targets -- -D warnings` and `cargo test` green.
3. Run `cargo fmt` before committing.
4. Performance-affecting changes to `src/engine.rs` or `src/speculative.rs`
   should include before/after numbers from `nirvana-code bench`.
5. Open a pull request with a short description of *why* — the diff already says *what*.

## Engine tests

Engine-level tests that need a real model are gated behind an environment variable
so the default `cargo test` stays fast and offline:

```sh
NIRVANA_TEST_MODEL=~/.nirvana/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
NIRVANA_TEST_DRAFT=~/.nirvana/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
cargo test --release -- --ignored
```

`NIRVANA_TEST_DRAFT` is optional (the target drafts for itself when unset).
`NIRVANA_TEST_PDF=/path/to/any.pdf` enables the PDF extraction test.

## Commit messages

Conventional-commit style prefixes are used: `feat:`, `fix:`, `perf:`, `docs:`,
`refactor:`, `test:`, `ci:`.

## Security

Please do not open public issues for security problems in the HTTP server.
Email the maintainer listed in `Cargo.toml` instead.
