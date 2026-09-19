# Nirvana Code

**A local coding assistant for Apple Silicon, built on llama.cpp's Metal backend.**
Terminal UI, web UI, and an OpenAI-compatible API in one Rust binary. No cloud, no Python in the main path.

```
nirvana-code download qwen-coder-3b     # 2.1 GB
nirvana-code run                        # terminal UI
nirvana-code web                        # browser UI at http://127.0.0.1:8080
nirvana-code serve                      # OpenAI-compatible API for editors and SDKs
```

## What it does for speed

Everything below is measured, not marketed. Numbers are from an M2 Pro (16 GB) with Qwen2.5-Coder-1.5B Q4_K_M unless noted; run `nirvana-code bench --json` to get your own.

| Mechanism | What it is | Measured effect |
|---|---|---|
| **Persistent prefix cache** | The KV cache lives across turns. Each request rolls back to the longest shared token prefix and evaluates only what changed. | Warm TTFT **37 ms** vs 552 ms cold on an 862-token prompt (93 % lower) |
| **KV state persistence** (`--persist-kv`) | The prefix KV state is written to disk on exit and restored on the next launch. | First-turn TTFT in a fresh process: **22 ms** vs 87 ms |
| **Speculative decoding** (`--draft-model`) | A small draft model proposes tokens; the target verifies them in one batched pass. Draft length adapts 1–16. Output is byte-identical to plain decoding. | Pays off when the target is 5–10× the draft (7B+). On a 1.5B target it is *slower* (81 vs 116 tok/s at 66 % acceptance) — the target is already faster than the draft overhead. |
| **Prompt-lookup decoding** (`--ngram-speculative`) | Self-speculation from n-gram repeats in the context; no second model. | Gains on repetitive edits/refactors; neutral otherwise (116.6 vs 115.8 tok/s on prose). |
| **Quantized KV cache** (`--kv-type q8_0`/`q4_0`) | 8- or 4-bit attention cache with Flash Attention on. | Halves / quarters KV memory so larger contexts and models fit in 16 GB. |
| **Memory fit** | llama.cpp's fitter picks the GPU offload split; `mlock` is auto-disabled above 70 % of RAM; when weights exceed the GPU-wired budget the exact `sysctl iogpu.wired_limit_mb` to raise it is printed. | 27B-class models load on 16 GB instead of thrashing. |
| **Model-native chat templates** | The GGUF's own Jinja template is rendered (Gemma, Llama 3, Mistral, DeepSeek, Qwen …), not a hard-coded ChatML. | Correct output from non-ChatML models. |

Baseline on this machine: **1,560 tok/s prefill, 114 tok/s decode** (1.5B Q4_K_M); 22 tok/s decode on a 9B Q6_K.

## Install

Requires macOS 14+ on Apple Silicon and the Xcode Command Line Tools (`xcode-select --install`).

```bash
# From source (recommended until the Homebrew tap is published)
git clone https://github.com/<you>/nirvana-code && cd nirvana-code
cargo install --path .           # uses target-cpu=native for this machine

# Homebrew (formula in packaging/nirvana-code.rb — fill in the release URL and sha256)
brew install --formula packaging/nirvana-code.rb
```

For a binary that runs on every M-series chip use `scripts/build-release.sh`, which builds against the M1 baseline and signs it.

## Models

`nirvana-code models` lists the catalog and what is installed. Downloads resume if interrupted and are verified against the SHA-256 that Hugging Face publishes for the file. Models found in `~/.lmstudio/models` are picked up automatically.

```bash
nirvana-code download qwen-0.5b          # 0.4 GB — draft model for speculative decoding
nirvana-code download qwen-coder-1.5b    # 1.0 GB
nirvana-code download qwen-coder-3b      # 2.1 GB
nirvana-code download qwen-3.5-9b        # 7.4 GB — best quality that leaves room on 16 GB
nirvana-code download qwen-3.8-27b       # 13.8 GB — needs the wired-limit sysctl on 16 GB
```

## Usage

```bash
nirvana-code run                                   # TUI (Ctrl+K palette, Ctrl+C stop, Ctrl+R clear)
nirvana-code prompt "Write a lock-free ring buffer in Rust"
nirvana-code prompt "…" --preset clean-refactor    # see `nirvana-code prompt --help` for presets
nirvana-code -m qwen-3.5-9b --draft-model qwen-0.5b run    # speculative decoding (same tokenizer family)
nirvana-code --persist-kv --ctx-size 8192 --kv-type q8_0 web
nirvana-code bench --runs 5 --json                 # reproducible numbers for issues and PRs
```

Persistent defaults go in `~/.config/nirvana-code/config.toml` (every long flag is a key, e.g. `ctx_size = 8192`); explicit flags override it.

Useful flags: `--seed N` for reproducible sampling, `--repeat-penalty 1.1`, `--dry-multiplier 0.8`, `--ubatch 1024` on 30+-GPU-core chips, `--verbose` to see llama.cpp's own logs.

## API server

`nirvana-code serve` exposes `POST /v1/chat/completions` (streaming and not), `GET /v1/models`, `POST /v1/models/load`, `POST /v1/chat/stop`, and works with the OpenAI SDKs, Continue, Cursor, and Neovim clients. Extra request fields: `min_p`, `top_k`, `seed`, `repeat_penalty`, `frequency_penalty`, `presence_penalty`, `dry_multiplier`, `ngram_speculative`.

### Security model

The server binds to `127.0.0.1` by default and is designed so that a web page open in your browser cannot use it against you:

- **No CORS headers** unless you pass `--cors-origin <origin>` (`*` for any). The bundled web UI is same-origin and needs none.
- **Host header check** rejects DNS-rebinding attempts; add LAN names with `--allow-host`.
- **`--api-key <key>`** (or `NIRVANA_API_KEY`) requires `Authorization: Bearer` on every `/v1/*` route. Binding to a non-loopback `--host` without a key generates one and prints it.
- **Project browsing** (`/v1/project/*`) reads only inside `--workspace <dir>`; `web` defaults it to the current directory, `serve` disables it unless set.
- **Model loading** by API accepts installed model names only, never filesystem paths.
- The Unix socket (`--socket`) is created `0600`.

## Backends

| Backend | Status | Notes |
|---|---|---|
| GGUF on Metal (llama.cpp) | Primary | Everything in the speed table above. |
| GGUF + draft GGUF | Stable | `--draft-model`; vocabularies must match (checked at load). |
| MLX | **Experimental** | Launches `python -m mlx_lm server` as a subprocess and talks to it over HTTP. Needs `pip install mlx-lm`; ~25 s startup; prefix caching is managed by mlx_lm, not by this project. Kept for people with MLX-only weights. |

## Attachments

PDFs, images (Vision OCR), documents and code files can be attached in the TUI (`/attach`) and web UI. Extraction uses Apple's PDFKit and Vision through a small Swift helper compiled at build time and embedded in the binary — no runtime compiler, no Python.

## Development

```bash
cargo clippy --all-targets -- -D warnings && cargo test
NIRVANA_TEST_MODEL=~/.nirvana/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
NIRVANA_TEST_DRAFT=~/.nirvana/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
cargo test --release -- --ignored     # engine tests: prefix reuse, prompt-lookup and speculative == plain greedy
```

See `CONTRIBUTING.md`, `CHANGELOG.md`, and `RELEASE_PLAN.md` (the review that drove the current state, with what is still open).

## License

MIT — see `LICENSE`.
