# Nirvana Code

<div align="center">

**The native, ultra-low latency Apple Silicon coding assistant and local inference engine.**  
*Terminal UI, Cyberpunk Web UI, and OpenAI-compatible API — packed into a single standalone Rust binary.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%2014%2B%20(Apple%20Silicon)-black?logo=apple)](https://apple.com)
[![Metal 3](https://img.shields.io/badge/GPU-Metal%203%20Unified%20Memory-orange)](https://developer.apple.com/metal/)
[![Rust](https://img.shields.io/badge/Rust-2024%20Edition-red?logo=rust)](https://www.rust-lang.org)
[![Version](https://img.shields.io/badge/Version-v0.3.0-green)](Cargo.toml)

```bash
nirvana-code download qwen-coder-3b     # Resumable, SHA-256 verified download
nirvana-code run                        # Fullscreen interactive Terminal UI
nirvana-code web                        # Modern browser interface with live GPU HUD
nirvana-code serve                      # OpenAI-compatible API for Cursor, Continue, & Neovim
```

</div>

---

## Table of Contents

- [Why Nirvana Code?](#why-nirvana-code)
- [How Nirvana Code Differs from Other Tools](#how-nirvana-code-differs-from-other-tools)
- [What It Does for Speed](#what-it-does-for-speed)
- [Dual-Engine Architecture (GGUF + Apple MLX)](#dual-engine-architecture-gguf--apple-mlx)
- [Multimodal & Native Attachment Pipeline](#multimodal--native-attachment-pipeline)
- [User Interfaces](#user-interfaces)
  - [Terminal UI (TUI)](#terminal-ui-tui)
  - [Cyberpunk Web UI](#cyberpunk-web-ui)
- [API Server & Security Model](#api-server--security-model)
- [Installation](#installation)
- [Usage & Commands](#usage--commands)
- [Configuration](#configuration)
- [Contributing & License](#contributing--license)

---

## Why Nirvana Code?

Most local LLM tools are built as generic, cross-platform wrappers: heavy Electron frontends, persistent background Go daemons, or Python runtimes with dozens of virtualenv dependencies. They treat your Mac like a generic Linux box with a discrete graphics card, ignoring the unique architectural superpowers of Apple Silicon.

**Nirvana Code was engineered from the metal up exclusively for macOS and Apple Silicon (M1/M2/M3/M4):**

- **Single Native Binary**: Everything — the inference engine, Metal kernels, TUI, Web UI, Swift Vision OCR pipeline, and HTTP/UDS server — is compiled into a single lean binary (< 15 MB). Zero Docker, zero Electron, zero Python in the critical path.
- **Unified Memory Zero-Copy**: Model weights and KV caches reside in unified memory accessible simultaneously by CPU and Metal GPU without PCI-e bus copying.
- **Instant Warm TTFT (< 40 ms)**: Persistent KV prefix caching rolls back to the longest shared token prefix across turns, cutting time-to-first-token by up to 93%.
- **Hardware-Aware Core Scheduling**: Asymmetric scheduling binds compute-heavy Metal dispatch and token sampling to Apple Silicon **Performance (P) cores**, offloading async file I/O and web requests to **Efficiency (E) cores**.

---

## How Nirvana Code Differs from Other Tools

Here is how Nirvana Code stacks up against common local LLM runners and interfaces:

| Feature / Capability | **Nirvana Code** | **Ollama** | **LM Studio** | **llama.cpp / llama-server** | **vLLM / TGI** |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Primary Architecture** | **Native Rust (Apple Silicon Metal 3)** | Go daemon + llama.cpp subprocess | Closed-source Electron app (~500MB) | Raw C++ CLI / HTTP server | Python / CUDA (Linux Datacenter) |
| **Runtime Overhead & RAM** | **< 30 MB idle**, single binary | Heavy background service | 500 MB – 1 GB+ idle (Chromium) | < 30 MB idle | Multi-GB Python daemon |
| **All-in-One Interfaces** | **TUI + Web UI + CLI + API** in one binary | CLI + API only (no built-in UI) | GUI only (no native TUI) | Web UI + CLI (basic) | API only |
| **Persistent Prefix Caching** | **Built-in & Automatic** (across turns) | Clears context on fresh calls | Session-limited | Supported (`--cont-batching`) | PagedAttention (server-side) |
| **Disk-Persisted KV Cache** | **Yes (`--persist-kv`)** (22ms warm start) | No | No | Experimental CLI flags | No |
| **Speculative Decoding** | **Confidence-Gated 0.5B Draft** (+8% on 7B) | Basic draft (unfiltered) | No | Supported via flags | Supported (GPU-heavy) |
| **Prompt-Lookup Speculation** | **Built-in (`--ngram-speculative`)** | No | No | Basic CLI flag | No |
| **Dual Backend (GGUF + MLX)** | **Yes** (Runs GGUF & MLX seamlessly) | GGUF only | GGUF & MLX (via GUI) | GGUF only | HF Transformers / vLLM |
| **LM Studio Model Discovery** | **Automatic** (`~/.lmstudio/models`) | No (custom blob store) | Native | No | No |
| **Native macOS OCR & Attachments**| **Embedded Swift (Vision + PDFKit)** | Requires external vision model | Vision models only | CLI scripts | No |
| **Apple Silicon Core Affinity** | **P-core compute / E-core I/O** | OS default | OS default | Thread flags only | N/A (Linux) |
| **Local Security Hardening** | **Same-origin CORS, Host checks, UDS** | Open localhost by default | Open localhost by default | Basic flags | Server-oriented |
| **Open Source** | **100% MIT** | Open source (Go/C++) | Proprietary / Closed Source | Open source (C++) | Open source (Python/C++) |

### Key Architectural Advantages

#### 1. Nirvana Code vs. Ollama
- **No Background Daemon**: Ollama runs an unmanaged system daemon that stays resident in memory. Nirvana Code runs strictly when you invoke it, and shuts down cleanly when you exit.
- **Inter-Turn Prefix Caching**: Ollama frequently evicts or recomputes full prompt prefixes when switching parameters or sessions. Nirvana Code keeps the longest shared token prefix alive in the Metal KV cache, dropping warm TTFT from 552 ms down to **37 ms**.
- **Cold Boot KV Persistence**: With `--persist-kv`, Nirvana Code dumps the token cache to disk on exit and re-maps it into unified memory on next boot, yielding a **22 ms** first-token response in a fresh process.
- **Self-Contained TUI & Web UI**: Ollama requires third-party web apps (Open WebUI, etc.) requiring Docker. Nirvana Code includes both an interactive Ratatui TUI and a zero-dependency Cyberpunk Web UI out of the box.

#### 2. Nirvana Code vs. LM Studio
- **No Electron Bloat**: LM Studio consumes 600MB–1.2GB of RAM just for its Chromium interface before loading a single model weight. Nirvana Code's UI runs inside your terminal or in any browser using vanilla Web components with sub-millisecond rendering.
- **SSH & Terminal First**: Nirvana Code operates headlessly over SSH, in tmux sessions, and inside Neovim/Cursor setups seamlessly.
- **Complementary Ecosystem**: If you already use LM Studio, Nirvana Code automatically scans and discovers all models in `~/.lmstudio/models`, allowing you to run them in the TUI or API without duplicating gigabytes of storage.

#### 3. Nirvana Code vs. llama.cpp
- **Developer Experience**: llama.cpp is an incredible foundational library, but its CLI requires remembering dozens of raw low-level flags. Nirvana Code provides model catalogs, verified resumable downloads, dynamic `/model` switching, interactive project file browsing, and Jinja-rendered native chat templates without manual prompt formatting.
- **Apple Vision & PDFKit Integration**: Nirvana Code includes embedded native Swift helpers that extract text from PDFs and run on-device Apple Vision OCR on images without needing external Python packages or Tesseract.

---

## What It Does for Speed

Everything below is measured directly with `nirvana-code bench --json` on an **Apple M2 Pro (16 GB)** using `Qwen2.5-Coder-1.5B Q4_K_M` and `Qwen2.5-Coder-7B Q4_K_M`:

| Mechanism | What It Is | Measured Real-World Effect |
|:---|:---|:---|
| **Persistent Prefix Cache** | The KV cache lives across conversation turns. Each request rolls back to the longest shared token prefix and evaluates only newly appended tokens. | Warm TTFT **37 ms** vs. 552 ms cold on an 862-token prompt (**93% reduction in latency**). |
| **KV State Persistence** (`--persist-kv`) | The prefix KV state is serialized to NVMe disk on exit and restored on next process launch. | First-turn TTFT in a brand-new process: **22 ms** vs. 87 ms. |
| **Confidence-Gated Speculation** (`--draft-model`) | A lightweight 0.5B draft model proposes tokens while confidence stays above 0.75; the 7B target model verifies the batch in one Metal pass. | **7B Q4_K_M: 38.7 vs. 35.7 tok/s (+8% speedup) at 82% acceptance rate.** Byte-identical output guaranteed. |
| **Prompt-Lookup Speculation** (`--ngram-speculative`) | Self-speculative decoding from n-gram repeats in context; requires no second model. | High gains on repetitive code refactors, boilerplate, and JSON output; neutral on prose. |
| **Quantized KV Cache** (`--kv-type q8_0` / `q4_0`) | Quantizes attention keys and values to 8-bit or 4-bit with Flash Attention enabled. | Cuts KV memory footprint by 50–75%, allowing 32k+ context windows to fit in 16 GB unified RAM. |
| **Memory Fit & Wired Limits** | Queries hardware limits via IOKit/sysctl; automatically disables `mlock` above 70% RAM to prevent OS swapping; calculates exact `sysctl iogpu.wired_limit_mb` parameters. | Enables 27B-class models to load and run stably on 16 GB machines without kernel panics. |
| **Model-Native Jinja Templates** | GGUF-embedded Jinja chat templates are parsed dynamically via `minijinja`. | Guaranteed correct formatting for Qwen, Gemma, Llama 3, DeepSeek-R1, and Mistral without manual prompt strings. |

### Baseline Measurements (M2 Pro 16 GB, 862-token prompt)

| Model | Prefill Speed | Decode Speed | Cold TTFT | Warm TTFT (Prefix Cached) |
|:---|:---:|:---:|:---:|:---:|
| **Qwen2.5-Coder-1.5B Q4_K_M** | 1,560 tok/s | 114 tok/s | 552 ms | **37 ms** |
| **Qwen2.5-Coder-7B Q4_K_M** | 347 tok/s | 35 tok/s | 2,483 ms | **124 ms** |
| **Qwen 3.5 9B Q6_K** | 280 tok/s | 22 tok/s | 3,100 ms | **180 ms** |

---

## Dual-Engine Architecture (GGUF + Apple MLX)

Nirvana Code bridges the two premier Apple Silicon local inference formats under one unified interface:

```
                  ┌──────────────────────────────────────────────┐
                  │            Nirvana Code Unified Core         │
                  │   (CLI  •  TUI  •  Web UI  •  OpenAI API)    │
                  └──────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
  ┌───────────────────────────────┐               ┌───────────────────────────────┐
  │   llama.cpp Metal 3 Engine    │               │       Apple MLX Engine        │
  │   - Direct Metal Shaders      │               │   - Native MLX Py-Subprocess  │
  │   - Quantized KV (Q8_0/Q4_0)  │               │   - Zero-Copy Unified Memory  │
  │   - Speculative Drafting      │               │   - Reasoning Delta Streaming │
  │   - Jinja Chat Templates      │               │   - Character-Safe Loop Guard │
  └───────────────────────────────┘               └───────────────────────────────┘
```

- **GGUF Metal Engine**: Directly linked via C++ FFI. Ultra-fast, zero-overhead, supports prefix caching, speculative decoding, and quantized KV caches.
- **Apple MLX Engine**: Seamlessly executes MLX models downloaded from Hugging Face or LM Studio. Features an asynchronous SSE bridge with character-boundary UTF-8 safety and repetition loop interception.

---

## Multimodal & Native Attachment Pipeline

Attach files in either the Terminal UI (`/attach`) or the Web UI (drag & drop / paperclip):

- **PDF Documents**: Parsed using native macOS PDFKit (`mdls` + `textutil` + embedded Swift helper). Zero external Python PDF dependencies.
- **Images & Diagrams**: Analyzed using Apple's built-in **Vision Framework** for high-accuracy local OCR text extraction.
- **Source Code & Markdown**: Ingested with smart UTF-8 character-safe windowing (`.chars().take(25000)`) preventing token context blowouts.

---

## User Interfaces

### Terminal UI (TUI)

Launch with `nirvana-code run`:
- **Keyboard-Driven Workflow**: `Enter` sends prompt; `Ctrl+C` cleanly cancels generation; `Ctrl+R` clears session.
- **Command Palette (`Ctrl+K`)**: Fast fuzzy action menu.
- **Interactive Model Switcher (`/model`)**: Switch models instantly without restarting the process.
- **Workspace Sidebar (`Ctrl+B`)**: View active files and context telemetry.
- **Response Copy (`Ctrl+O`)**: Copy formatted markdown response or entire chat to macOS clipboard.

### Cyberpunk Web UI

Launch with `nirvana-code web` (served at `http://127.0.0.1:8080`):
- **Live Silicon HUD**: Displays active model, GPU offload layers, unified RAM utilization, decode speed (tok/s), and TTFT.
- **Real-Time Context Gauge**: Visual progress bar tracking token context consumption against the maximum window (`used / capacity`).
- **Collapsible Muted Thinking**: Dedicated `<think>` blocks for DeepSeek-R1 and reasoning models rendered in muted typography with expand/collapse toggles.
- **Workspace Split**: Dedicated sidebar tabs separating **Conversations** (multi-turn chat sessions) from **Projects** (codebase workspace browsing).
- **Safety Reset (`🛑 Reset`)**: Instantly aborts hung jobs, severs runaway generation streams, and releases worker locks.

---

## API Server & Security Model

`nirvana-code serve` exposes an OpenAI-compatible HTTP server (`/v1/chat/completions`, `/v1/models`, `/v1/models/load`, `/v1/chat/stop`) designed to integrate directly with **Cursor**, **Continue.dev**, and **Neovim**.

### Hardened Local Security
- **Strict Same-Origin CORS**: Rejects external web origins by default; browser tabs cannot make unauthorized cross-site requests to your local model.
- **Host Header Verification**: Blocks DNS-rebinding attacks targeting `127.0.0.1`.
- **Bearer Token Authentication**: Enforce with `--api-key <secret>` or `NIRVANA_API_KEY`.
- **Workspace Sandbox**: File browsing endpoints (`/v1/project/*`) are strictly locked within the `--workspace <dir>` boundary.
- **UNIX Domain Socket Support**: Run with `--socket /path/to/socket` with `0600` permissions for zero-network inter-process communication.

---

## Installation

### Prerequisites
- macOS 14.0 (Sonoma) or newer on Apple Silicon (M1/M2/M3/M4, Pro, Max, or Ultra).
- Xcode Command Line Tools: `xcode-select --install`.

### Option 1: From Source (Recommended)
```bash
git clone https://github.com/niravlekinwala/nirvana-code.git
cd nirvana-code
cargo build --release
# Copy binary to your PATH:
cp target/release/nirvana-code /usr/local/bin/
```

### Option 2: Homebrew
```bash
brew tap niravlekinwala/nirvana
brew install nirvana-code

# Or install in a single command:
brew install niravlekinwala/nirvana/nirvana-code
```
> **Note for macOS Homebrew 4.4+**: If prompted with `Refusing to load formula from untrusted tap`, run `brew trust niravlekinwala/nirvana`.

---

## Usage & Commands

```bash
# Model Management
nirvana-code models                                # List installed and catalog models
nirvana-code download qwen-coder-1.5b              # Download 1.5B coder model (1.0 GB)
nirvana-code download qwen-coder-7b                # Download 7B coder model (4.7 GB)
nirvana-code download qwen-0.5b                    # Download 0.5B draft model (0.4 GB)

# Interactive Modes
nirvana-code run                                   # Launch fullscreen Terminal UI (TUI)
nirvana-code web                                   # Launch Web UI on http://127.0.0.1:8080
nirvana-code serve --port 8080                     # Start headless OpenAI API server

# Single-Shot CLI Prompts
nirvana-code prompt "Write a lock-free ring buffer in Rust"
nirvana-code prompt "Explain this code" --preset clean-refactor

# Advanced Performance Flags
nirvana-code -m qwen-coder-7b --draft-model qwen-0.5b run   # Speculative decoding
nirvana-code --persist-kv --ctx-size 8192 --kv-type q8_0 web # Quantized KV + persistence
nirvana-code bench --runs 5 --json                          # Run reproducible benchmark
```

---

## Configuration

Persistent options can be defined in `~/.config/nirvana-code/config.toml`. Any CLI flag can be mapped directly:

```toml
# ~/.config/nirvana-code/config.toml
model = "qwen-coder-7b"
ctx_size = 8192
kv_type = "q8_0"
persist_kv = true
temperature = 0.2
top_p = 0.95
min_p = 0.05
repeat_penalty = 1.1
dry_multiplier = 0.8
web_port = 8080
```

CLI flags override settings defined in `config.toml`.

---

## Contributing & License

Contributions, benchmark results from different Apple Silicon chips, and feature requests are welcome! Please check out [CONTRIBUTING.md](CONTRIBUTING.md) and review [CHANGELOG.md](CHANGELOG.md).

Distributed under the **MIT License**. See [LICENSE](LICENSE) for full details.
