use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "nirvana-code",
    about = "⚡ Nirvana Code: Ultra-low latency Apple Silicon coding assistant & prompt engineering engine with Metal 3, Q8_0 KV-Cache, Prefix Caching, and Speculative Decoding.",
    version = env!("CARGO_PKG_VERSION")
)]
pub struct Cli {
    #[arg(
        short = 'm',
        long = "model",
        global = true,
        help = "Path to GGUF model file or catalog ID"
    )]
    pub model: Option<PathBuf>,

    #[arg(
        long = "draft-model",
        global = true,
        help = "Path to draft model for speculative decoding"
    )]
    pub draft_model: Option<PathBuf>,

    #[arg(
        long = "speculative",
        global = true,
        help = "Enable speculative decoding engine"
    )]
    pub speculative: bool,

    #[arg(
        long = "n-draft",
        global = true,
        default_value = "4",
        help = "Initial draft length for speculative decoding (adapts 1-16 at runtime)"
    )]
    pub n_draft: usize,

    #[arg(
        long = "verbose",
        short = 'v',
        global = true,
        help = "Keep llama.cpp diagnostics on stderr"
    )]
    pub verbose: bool,

    #[arg(
        long = "workspace",
        global = true,
        help = "Directory the server's project endpoints may read (web: defaults to cwd; serve: disabled unless set)"
    )]
    pub workspace: Option<PathBuf>,

    #[arg(
        long = "api-key",
        global = true,
        env = "NIRVANA_API_KEY",
        help = "Require 'Authorization: Bearer <key>' on /v1/* (auto-generated when binding beyond loopback)"
    )]
    pub api_key: Option<String>,

    #[arg(
        long = "cors-origin",
        global = true,
        help = "Origin allowed to call the API from a browser (repeatable; '*' allows any). Default: same-origin only"
    )]
    pub cors_origin: Vec<String>,

    #[arg(
        long = "allow-host",
        global = true,
        help = "Additional Host header values to accept (e.g. a LAN hostname); repeatable"
    )]
    pub allow_host: Vec<String>,

    #[arg(
        long = "kv-type",
        global = true,
        default_value = "f16",
        help = "KV-Cache quantization format [f16, q8_0, q4_0, auto]"
    )]
    pub kv_type: String,

    #[arg(
        long = "no-mlock",
        global = true,
        help = "Disable unified RAM memory locking (mlock enabled by default)"
    )]
    pub no_mlock: bool,

    #[arg(
        short = 'g',
        long = "gpu-layers",
        global = true,
        default_value = "99",
        help = "Number of layers to offload to Apple Silicon Metal GPU"
    )]
    pub gpu_layers: u32,

    #[arg(
        short = 'c',
        long = "ctx-size",
        global = true,
        default_value = "4096",
        help = "Context token capacity"
    )]
    pub ctx_size: u32,

    #[arg(
        long = "max-tokens",
        global = true,
        default_value = "2048",
        help = "Maximum output tokens to generate"
    )]
    pub max_tokens: usize,

    #[arg(
        short = 't',
        long = "temperature",
        global = true,
        default_value = "0.7",
        help = "Sampling temperature"
    )]
    pub temperature: f32,

    #[arg(
        long = "min-p",
        global = true,
        default_value = "0.05",
        help = "Minimum P sampling threshold (cuts low-probability syntax hallucinations)"
    )]
    pub min_p: f32,

    #[arg(
        long = "top-p",
        global = true,
        default_value = "0.9",
        help = "Nucleus sampling threshold"
    )]
    pub top_p: f32,

    #[arg(
        long = "top-k",
        global = true,
        default_value = "40",
        help = "Top-K sampling limit"
    )]
    pub top_k: i32,

    #[arg(
        long = "repeat-penalty",
        global = true,
        default_value = "1.0",
        help = "Repetition penalty over the last 64 tokens (1.0 = off, 1.1 = mild)"
    )]
    pub repeat_penalty: f32,

    #[arg(
        long = "dry-multiplier",
        global = true,
        default_value = "0.0",
        help = "DRY anti-repetition multiplier (0 = off, 0.8 = typical)"
    )]
    pub dry_multiplier: f32,

    #[arg(
        long = "ubatch",
        global = true,
        help = "Physical micro-batch size for prefill (default 512; try 1024 on 30+ GPU-core chips)"
    )]
    pub ubatch: Option<u32>,

    #[arg(
        long = "persist-kv",
        global = true,
        help = "Save the KV state of the prefix (up to 1024 tokens) on exit and restore it next start"
    )]
    pub persist_kv: bool,

    #[arg(
        long = "seed",
        global = true,
        help = "Sampler RNG seed for reproducible output (random per generation if unset)"
    )]
    pub seed: Option<u32>,

    #[arg(
        long = "ngram-speculative",
        global = true,
        help = "Enable prompt lookup decoding (self-speculative n-gram matching) for 1.5x-2x code speedup"
    )]
    pub ngram_speculative: bool,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    #[command(about = "Launch interactive Nirvana Code Cyberpunk terminal")]
    Run,

    #[command(about = "Launch interactive offline coding assistant")]
    Chat,

    #[command(about = "List installed models and AtomicChat 16GB guide catalog")]
    Models,

    #[command(about = "Download a model by catalog ID, filename, or URL")]
    Download {
        #[arg(help = "Catalog model ID (e.g. 'qwen-3.8-27b', 'qwen-3.5-9b') or direct URL")]
        target: String,
    },

    #[command(about = "Benchmark prefill, cold/warm TTFT and decode speed (median of N runs)")]
    Bench {
        #[arg(
            short = 'n',
            long = "num-tokens",
            default_value = "128",
            help = "Number of output tokens per run"
        )]
        num_tokens: usize,

        #[arg(
            long = "runs",
            default_value = "5",
            help = "Number of measured runs (after one warm-up)"
        )]
        runs: usize,

        #[arg(
            long = "prompt-tokens",
            default_value = "512",
            help = "Approximate prompt length in tokens"
        )]
        prompt_tokens: usize,

        #[arg(long = "json", help = "Emit machine-readable JSON instead of a table")]
        json: bool,
    },

    #[command(about = "Single-shot headless generation or prompt optimization")]
    Prompt {
        #[arg(help = "The input prompt or code question")]
        prompt: String,
        #[arg(
            short = 'p',
            long = "preset",
            default_value = "offline-assistant",
            help = "Template preset ID (e.g. claude-37-hybrid, antigravity-20, deepseek-r1)"
        )]
        preset: String,
    },

    #[command(about = "Start OpenAI-compatible HTTP API server & Unix Domain Socket daemon")]
    Serve {
        #[arg(
            short = 'p',
            long = "port",
            default_value = "8080",
            help = "HTTP port to bind to"
        )]
        port: u16,

        #[arg(
            long = "host",
            default_value = "127.0.0.1",
            help = "Host address to bind to"
        )]
        host: String,

        #[arg(
            long = "socket",
            help = "Unix domain socket path (e.g. /tmp/nirvana.sock)"
        )]
        socket: Option<PathBuf>,
    },

    #[command(about = "Start Nirvana Code Web UI server and launch in browser")]
    Web {
        #[arg(
            short = 'p',
            long = "port",
            default_value = "8080",
            help = "HTTP port to bind to"
        )]
        port: u16,

        #[arg(
            long = "host",
            default_value = "127.0.0.1",
            help = "Host address to bind to"
        )]
        host: String,

        #[arg(long = "no-open", help = "Do not automatically open the browser")]
        no_open: bool,
    },
}

impl Cli {
    pub fn kv_mode(&self) -> crate::engine::KvQuantMode {
        use crate::engine::KvQuantMode;
        match self.kv_type.to_lowercase().as_str() {
            "q4_0" => KvQuantMode::Q4_0,
            "f16" => KvQuantMode::F16,
            "q8_0" => KvQuantMode::Q8_0,
            _ => KvQuantMode::Auto,
        }
    }
}
