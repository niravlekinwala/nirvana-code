use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "nirvana-code",
    about = "⚡ Nirvana Code: Ultra-low latency Apple Silicon coding assistant & prompt engineering engine with Metal 3, Q8_0 KV-Cache, Prefix Caching, and Speculative Decoding.",
    version = env!("CARGO_PKG_VERSION")
)]
pub struct Cli {
    #[arg(short = 'm', long = "model", global = true, help = "Path to GGUF model file or catalog ID")]
    pub model: Option<PathBuf>,

    #[arg(long = "draft-model", global = true, help = "Path to draft model for speculative decoding")]
    pub draft_model: Option<PathBuf>,

    #[arg(long = "speculative", global = true, help = "Enable speculative decoding engine")]
    pub speculative: bool,

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

    #[command(about = "Run Apple Silicon benchmark (TTFT with/without prefix cache, TPS)")]
    Bench {
        #[arg(
            short = 'n',
            long = "num-tokens",
            default_value = "128",
            help = "Number of benchmark tokens to generate"
        )]
        num_tokens: usize,
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

        #[arg(
            long = "no-open",
            help = "Do not automatically open the browser"
        )]
        no_open: bool,
    },
}
