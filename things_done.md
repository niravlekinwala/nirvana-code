# Nirvana Code: Progress & Feature Summary

This document summarizes the features, architectural enhancements, hardware optimizations, and reliability fixes implemented in **Nirvana Code** to date.

---

## 1. Dual-Backend Architecture (GGUF + Apple MLX)

- **Apple MLX Engine Integration ([`src/mlx_engine.rs`](file:///Users/nirav/GitHubRepositories/nirvana-code/src/mlx_engine.rs))**:
  - Direct runtime support for native Apple MLX models alongside llama.cpp GGUF.
  - Automatic zero-overhead subprocess lifecycle management with HTTP/SSE streaming.
  - Full compatibility with reasoning models (DeepSeek-R1, Gemma 3, Qwen 2.5).
- **Multi-Directory Model Discovery ([`src/model_manager.rs`](file:///Users/nirav/GitHubRepositories/nirvana-code/src/model_manager.rs))**:
  - Automatic scanning across `~/.nirvana/models`, `~/.lmstudio/models`, and local `./models` directories.
  - Instant detection and categorization of both GGUF formats and MLX directories (containing `config.json`).
- **Interactive Model Selection (`/model`)**:
  - In-chat command (`/model`) and UI modal to browse, inspect, and switch active models on the fly.
  - Clean labeling distinguishing Apple MLX from Metal 3 GGUF without redundant tags.

---

## 2. Multimodal & Attachment Pipeline ([`src/attachment.rs`](file:///Users/nirav/GitHubRepositories/nirvana-code/src/attachment.rs))

- **Universal Attachment Ingestion**:
  - Support for dragging, dropping, or selecting code files, plaintext, images, and PDF documents.
- **Native macOS Zero-Dependency PDF Extraction**:
  - Utilizes native macOS system utilities (`mdls`, `sips`, `textutil`, Python fallbacks) to parse PDF content and metadata cleanly without heavy external PDF runtimes.
- **Base64 & Local Path Resolvers**:
  - Flexible ingestion accepting both client-uploaded base64 payloads and direct filesystem file paths.
- **Safe Truncation Handling**:
  - Unicode character-safe text truncation (`.chars().take(25000)`) preventing UTF-8 boundary slicing when processing large documents into prompt context.

---

## 3. Web UI & Workspace Features ([`src/web/index.html`](file:///Users/nirav/GitHubRepositories/nirvana-code/src/web/index.html))

- **Sidebar Separation (Conversations vs. Projects)**:
  - **Conversations**: Standard multi-turn chat sessions with persistent local history.
  - **Projects**: Dedicated project view tailored for working directly on local codebase directories and files.
- **Live Context Window Gauge**:
  - Real-time progress bar displaying current context token usage vs. total context window capacity (`context_used / context_capacity`).
- **Collapsible Muted Thinking Display**:
  - Reasoning blocks (`<think>...</think>`) formatted in muted color styling with expand/collapse toggles, keeping main responses clean and legible.
- **Force Unstick & Reset Mechanism**:
  - One-click safety button (`🛑 Reset / Unstick`) in the UI to abort hanging generations, cancel in-flight HTTP requests, and free backend worker locks.

---

## 4. Hardware-Level Optimizations & Stability Fixes

- **Batch Size Assertion Fix ([`src/engine.rs`](file:///Users/nirav/GitHubRepositories/nirvana-code/src/engine.rs))**:
  - Resolved `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` crash by implementing prompt chunking and sliding batch evaluation for long inputs.
- **UTF-8 Slicing Panic Elimination ([`src/mlx_engine.rs`](file:///Users/nirav/GitHubRepositories/nirvana-code/src/mlx_engine.rs))**:
  - Replaced byte-level string slicing in the repetition detector with a character-based Unicode scalar array (`Vec<char>`).
  - Completely eliminated mid-stream panics triggered by multi-byte UTF-8 sequences (e.g., curly quotes `’`, emojis `🚀`, non-ASCII symbols).
- **Client & Server Repetition Degeneration Loop Detection**:
  - Real-time detection of repeating token degenerate loops with automatic graceful halting.
- **Apple Silicon Hardware Tuning**:
  - Metal 3 Unified Memory zero-copy inference.
  - ARM NEON SIMD vectorization and cache-line alignment.
  - Distinction between Performance (P) and Efficiency (E) core thread pools.

---

## 5. Current Test & Build Status

| Component | Status | Details |
| :--- | :---: | :--- |
| **Unit Tests** | Passed | 9/9 unit tests passing (`cargo test`) |
| **Debug Binary** | Compiled | `target/debug/nirvana-code` |
| **Release Binary** | Compiled | `target/release/nirvana-code` |
| **Port 8080** | Clean | Ready for manual start |

