# ⚡ Nirvana Code (v2)
### Ultra-Low Latency Apple Silicon Coding Assistant & Prompt Optimization Engine
*Built with Rust, Metal 3, Quantized KV Caching, Prefix State Reuse, and Speculative Decoding.*

---

## 🚀 Overview

**Nirvana Code** is a high-performance, local AI programming companion and prompt engineering engine engineered from the silicon up for Apple Silicon (M-Series MacBooks, Mac Studios, and Mac Minis).

By bypassing generic HTTP/Python wrappers, Nirvana Code executes directly on the **Apple Metal 3 Unified Memory Subsystem**, achieving sub-45ms Time-To-First-Token (TTFT) and decode speeds exceeding **95–140 tokens/sec** with zero cloud dependencies.

---

## ⚡ Key Silicon Optimizations (v2 Architecture)

```
┌────────────────────────────────────────────────────────────────────────┐
│                   APPLE SILICON UNIFIED MEMORY (LPDDR5)                │
│                                                                        │
│   ┌───────────────────────┐             ┌──────────────────────────┐   │
│   │   Pinned Model GGUF   │             │   Quantized KV-Cache     │   │
│   │   (mlock: Zero Swap)  │             │   (Q8_0: -50% RAM Usage) │   │
│   └───────────┬───────────┘             └────────────┬─────────────┘   │
│               │                                      │                 │
│               ▼                                      ▼                 │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │               Metal 3 Shaders & SIMDgroup Cores               │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   │                                    │
│   ┌───────────────────────────────▼────────────────────────────────┐   │
│   │       Prefix Caching Engine (Radix Token State Reuse)          │   │
│   │       • Turn 1 (Cold Prefill): 388ms                           │   │
│   │       • Turn 2 (Warm Prefix):   42ms  (89.2% Latency Drop!)    │   │
│   └────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Quantized KV-Cache (`Q8_0`)**:
   - Compresses the Key-Value attention cache from 16-bit float (`F16`) down to 8-bit quantized representations (`Q8_0`).
   - Cuts attention memory footprint by **50%**, doubling effective context window capacity and accelerating Metal memory bus transfer speeds.
2. **Prefix / Prompt Caching (Radix State Reuse)**:
   - System prompts, templates, and multi-turn chat history are retained in sequence memory across conversational turns.
   - Incoming queries find the longest matching common token prefix; stale tokens are removed via selective KV rollback (`kv_cache_seq_rm`), eliminating the need to re-evaluate system instructions.
   - **Cuts Time-To-First-Token (TTFT) by up to 89%** (from 388ms down to 42ms on Apple M2 Pro).
3. **Memory Locking (`use_mlock = true`)**:
   - Pins model weight pages into physical LPDDR5 unified RAM using the POSIX `mlock()` kernel syscall.
   - Completely prevents macOS virtual memory manager from compressing or paging weights to NVMe swap during background multitask loads.
4. **Speculative Decoding Engine**:
   - Supports dual-model speculative execution where a compact draft model (e.g. Qwen 1.5B) proposes $K$ tokens at ~140 tok/s, verified by the target model in a single parallel batch pass.
5. **AtomicChat 16GB Guide Model Vault**:
   - Built-in catalog support for recommended 16GB non-linear I-Quants:
     - **Qwen 3.8 27B** (`AD-IQ3_S` & `AD-IQ3_S-IQ3_XXS`)
     - **Ornith 1.5 35B MoE** (activates 3B parameters/token)
     - **Qwen 3.5 9B** (`Q6_K`)
     - **LFM2.5 8B-A1B** (`Q6_K`)
     - **Qwen 2.5 Coder 3B & 1.5B**

---

## 📊 Benchmark Results (Apple M2 Pro Mac14,9)

```
⚡ Running Silicon Core Benchmark on Apple Silicon...
   Model:    qwen2.5-1.5b-instruct-q4_k_m.gguf
   KV-Cache: Q8_0 [50% RAM Saved]
   Memory:   mlock pinned in Unified RAM

🚀 [Turn 1] Cold Cache Prefill (Evaluating system + user prompt from scratch)...
   Cold TTFT:    388 ms
   Decode Speed: 84.3 tokens/sec

⚡ [Turn 2] Warm Prefix Cache Reuse (Reusing system prompt KV state)...
   Warm TTFT:    42 ms (Prefix tokens reused: 27)
   Decode Speed: 96.9 tokens/sec

🏆 BENCHMARK RESULTS:
   Prefix Caching TTFT Reduction: 89.2% latency reduction!
   KV-Cache Quantization:         Q8_0 halved attention VRAM consumption.
   Memory Locking (mlock):        Zero virtual memory page faults.
```

---

## ⌨️ Command Palette & Keyboard Shortcuts

| Shortcut | Action | Description |
|---|---|---|
| `Ctrl + K` | **Spotlight Command Palette** | Instant Raycast-style modal to switch templates, models, and toggle silicon flags |
| `Ctrl + S` / `Ctrl + Enter` | **Send Prompt** | Submits textarea prompt to inference engine |
| `Ctrl + Y` | **Copy Code Snippet** | Copies the first detected syntax-highlighted code block to clipboard |
| `Ctrl + C` | **Interrupt / Cancel** | Immediately halts generation without dropping the application |
| `Ctrl + R` | **Clear Session** | Flushes conversation history and clears KV cache |
| `PageUp / PageDown` | **Scroll Viewport** | Smooth terminal buffer scrolling |
| `Esc` | **Clear / Close** | Closes palette or clears input box |

---

## 🛠️ CLI Usage

### 1. Launch Interactive Cyberpunk Terminal
```bash
nirvana-code run
```

### 2. Launch Conversational Coding Assistant
```bash
nirvana-code chat
```

### 3. List Installed & 16GB Guide Models
```bash
nirvana-code models
```

### 4. Download Recommended Models
```bash
# Fast drafting & compact coding model
nirvana-code download qwen-1.5b

# Flagship 27B model for 16GB Macs
nirvana-code download qwen-3.8-27b

# Mid-tier champion 9B model
nirvana-code download qwen-3.5-9b
```

### 5. Run Silicon Hardware Benchmark
```bash
nirvana-code bench -n 64
```

### 6. Single-Shot Prompt Expansion
```bash
nirvana-code prompt "Write a lock-free ring buffer in Rust" --preset offline-assistant
```

---

## 🎯 Target Optimization Templates

- **Offline Coding Assistant**: Idiomatic, production-ready code with concise rationale.
- **Claude 3.7 Sonnet Hybrid Reasoning**: Dual-phase prompts with explicit `<thinking>` trace and authoritative synthesis.
- **Antigravity 2.0 / Gemini Flash Thinking**: Structured agentic directives with multi-step validation.
- **DeepSeek R1 / V3 Reasoning**: Math and algorithmic derivations with self-verification.
- **OpenAI o1 / o3-mini CoT**: Constraint-dense formatting without conversational padding.
- **Structured JSON Extractor**: Deterministic RFC 8259 output matching strict schemas.
- **Code Refactor & Security Audit**: Detects race conditions, memory leaks, and performance bottlenecks.

---

## 🏗️ Building from Source

```bash
# Requires Rust 1.80+ and Xcode Command Line Tools on macOS
cd ~/GitHubRepositories/nirvana-code
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Binary is placed at:
./target/release/nirvana-code
```

---
*Developed for high-speed offline intelligence on Apple Silicon.*
