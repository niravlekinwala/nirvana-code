mod app;
pub mod attachment;
mod chat;
mod cli;
mod clipboard;
mod config;
mod engine;
#[cfg(test)]
mod engine_tests;
mod hardware;
pub mod mlx_engine;
mod model_manager;
mod palette;
mod server;
mod speculative;
mod syntax;
mod templates;
mod theme;
mod ui;

use anyhow::{Context, Result};
use app::{App, EngineState};
use cli::{Cli, Commands};
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers, MouseEventKind,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use engine::{GenerationConfig, InferenceEngine, KvQuantMode, StreamEvent};
use model_manager::{MODEL_CATALOG, ModelManager};
use palette::PaletteManager;
use ratatui::{Terminal, backend::CrosstermBackend};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};
use templates::TEMPLATES;
use tokio::sync::mpsc::unbounded_channel;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = config::parse_cli();
    engine::set_verbose(cli.verbose);

    match &cli.command {
        Some(Commands::Models) => {
            cmd_list_models()?;
            return Ok(());
        }
        Some(Commands::Download { target }) => {
            ModelManager::download_target(target).await?;
            return Ok(());
        }
        Some(Commands::Bench {
            num_tokens,
            runs,
            json,
            prompt_tokens,
        }) => {
            cmd_benchmark(&cli, *num_tokens, *runs, *json, *prompt_tokens).await?;
            return Ok(());
        }
        Some(Commands::Prompt { prompt, preset }) => {
            cmd_single_shot(&cli, prompt, preset).await?;
            return Ok(());
        }
        Some(Commands::Serve { port, host, socket }) => {
            cmd_serve(&cli, *port, host, socket.as_deref()).await?;
            return Ok(());
        }
        Some(Commands::Web {
            port,
            host,
            no_open,
        }) => {
            cmd_web(&cli, *port, host, !*no_open).await?;
            return Ok(());
        }
        _ => {}
    }

    // Resolve model path
    let model_path = match ModelManager::resolve_model_path(cli.model.as_deref()) {
        Some(p) => p,
        None => {
            eprintln!("\n❌ No installed model found!");
            eprintln!("Run the following command to download the recommended model:");
            eprintln!("   nirvana-code download qwen-1.5b\n");
            eprintln!("Or download a 16GB flagship model from the AtomicChat guide:");
            eprintln!("   nirvana-code download qwen-3.8-27b");
            eprintln!("   nirvana-code download qwen-3.5-9b\n");
            return Ok(());
        }
    };

    let kv_mode = cli.kv_mode();

    let use_mlock = !cli.no_mlock;

    let is_mlx = ModelManager::is_mlx_model(&model_path);
    let backend_str = if is_mlx {
        "Apple Silicon MLX (Unified Memory GPU)"
    } else {
        "Metal 3 GPU (llama.cpp)"
    };

    println!("⚡ Loading Nirvana Code Silicon Engine...");
    println!("   Model:       {}", model_path.display());
    println!("   Backend:     {backend_str}");
    println!("   KV-Cache:    {}", kv_mode.label());
    println!(
        "   MLock:       {}",
        if use_mlock {
            "Enabled (LPDDR5 RAM Pinned)"
        } else {
            "Disabled"
        }
    );
    if !is_mlx {
        println!("   Metal GPU:   {} layers offloaded", cli.gpu_layers);
    }
    println!("   Context:     {} tokens", cli.ctx_size);
    if cli.ngram_speculative {
        println!("   Speculation: Prompt Lookup Decoding (N-gram matching) ENABLED");
    }
    println!();

    let engine = load_engine(&cli, &model_path, kv_mode)?;
    println!("   Chat format: {}", engine.chat_format_label());

    run_tui(
        engine,
        model_path,
        cli.max_tokens,
        cli.temperature,
        cli.min_p,
        cli.top_p,
        cli.top_k,
        cli.ngram_speculative,
        cli.seed,
        cli.persist_kv,
    )?;
    Ok(())
}

/// Load the configured backend: MLX for MLX directories, GGUF otherwise, and
/// GGUF + draft when `--draft-model` or `--speculative` is given.
fn load_engine(cli: &Cli, model_path: &Path, kv_mode: KvQuantMode) -> Result<InferenceEngine> {
    engine::set_ubatch(cli.ubatch);
    let engine = if ModelManager::is_mlx_model(model_path)
        || !(cli.speculative || cli.draft_model.is_some())
    {
        InferenceEngine::load(
            model_path,
            cli.gpu_layers,
            !cli.no_mlock,
            kv_mode,
            cli.ctx_size,
        )?
    } else {
        let draft_path = ModelManager::resolve_model_path(cli.draft_model.as_deref())
            .or_else(|| ModelManager::resolve_model_path(Some(Path::new("qwen-0.5b"))))
            .context("No draft model found for speculative decoding. Run 'nirvana-code download qwen-0.5b'")?;
        println!(
            "   Draft model: {} (speculative decoding, K adapts 1–16)",
            draft_path.display()
        );
        InferenceEngine::load_speculative(
            model_path,
            &draft_path,
            cli.gpu_layers,
            !cli.no_mlock,
            kv_mode,
            cli.ctx_size,
            cli.n_draft,
        )?
    };
    if cli.persist_kv {
        match engine.load_session() {
            Ok(0) => {}
            Ok(n) => println!("   KV state:    restored {n} prefix tokens from disk"),
            Err(e) => eprintln!("   KV state:    could not restore ({e})"),
        }
    }
    Ok(engine)
}

/// `--persist-kv`: write the prefix state so the next start is warm.
fn persist_session(cli: &Cli, engine: &InferenceEngine) {
    if !cli.persist_kv {
        return;
    }
    match engine.save_session() {
        Ok(0) => {}
        Ok(n) => eprintln!("   KV state:    saved {n} prefix tokens"),
        Err(e) => eprintln!("   KV state:    save failed ({e})"),
    }
}

/// Assemble server options from the CLI. `web` mode defaults the workspace to
/// the current directory (the UI's Projects tab needs one); `serve` requires an
/// explicit `--workspace`. Binding beyond loopback without `--api-key`
/// generates one and prints it.
fn server_options(
    cli: &Cli,
    host: &str,
    port: u16,
    socket_path: Option<PathBuf>,
    kv_mode: KvQuantMode,
    web_mode: bool,
) -> Result<server::ServerOptions> {
    let workspace =
        match &cli.workspace {
            Some(w) => Some(w.canonicalize().with_context(|| {
                format!("--workspace {}: not a readable directory", w.display())
            })?),
            None if web_mode => std::env::current_dir().ok(),
            None => None,
        };

    let loopback = matches!(host, "127.0.0.1" | "localhost" | "::1");
    let api_key = match (&cli.api_key, loopback) {
        (Some(k), _) => Some(k.clone()),
        (None, true) => None,
        (None, false) => {
            let key = generate_api_key();
            eprintln!(
                "⚠️  Binding to {host} (not loopback) with no --api-key; generated one for this session."
            );
            Some(key)
        }
    };

    let mut allowed_hosts = vec![
        "localhost".to_string(),
        "127.0.0.1".to_string(),
        "::1".to_string(),
    ];
    if !loopback {
        allowed_hosts.push(host.to_string());
    }
    allowed_hosts.extend(cli.allow_host.iter().cloned());

    Ok(server::ServerOptions {
        host: host.to_string(),
        port,
        socket_path,
        persist_kv: cli.persist_kv,
        gpu_layers: cli.gpu_layers,
        use_mlock: !cli.no_mlock,
        kv_mode,
        ctx_size: cli.ctx_size,
        security: server::Security {
            workspace,
            api_key,
            cors_origins: cli.cors_origin.clone(),
            allowed_hosts,
        },
    })
}

fn generate_api_key() -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut out = String::with_capacity(40);
    for _ in 0..3 {
        let v = std::collections::hash_map::RandomState::new()
            .build_hasher()
            .finish();
        out.push_str(&format!("{v:016x}"));
    }
    format!("nv-{out}")
}

fn cmd_list_models() -> Result<()> {
    println!("\n⚡ [NIRVANA CODE] Silicon Model Manager (Apple M2 Pro 16GB Unified RAM)\n");

    let installed = ModelManager::list_installed();
    println!("📦 Installed Local Models (Checked ~/.nirvana, ~/.lmstudio, ~/.promptcraft):");
    if installed.is_empty() {
        println!("   (None installed yet. Run 'nirvana-code download qwen-1.5b' to download)");
    } else {
        for (i, (path, name, size)) in installed.iter().enumerate() {
            let size_mb = size / (1024 * 1024);
            let size_str = if size_mb >= 1024 {
                format!("{:.1} GB", size_mb as f64 / 1024.0)
            } else {
                format!("{size_mb} MB")
            };
            let backend_badge = if ModelManager::is_mlx_model(path) {
                "[Apple MLX]"
            } else {
                "[Metal GGUF]"
            };
            println!(
                "   {:>2}. {:<12} {:<45} {:>8}   {}",
                i + 1,
                backend_badge,
                name,
                size_str,
                path.display()
            );
        }
    }

    println!("\n🌐 AtomicChat 16GB Hardware Guide Models (Ready to Download):");
    println!("   ─────────────────────────────────────────────────────────────────────────────");
    for meta in MODEL_CATALOG {
        let is_inst = installed.iter().any(|(_, name, _)| name == meta.filename);
        let status = if is_inst {
            "✔ INSTALLED"
        } else {
            "  AVAILABLE"
        };
        println!(
            "   [{}] {:<18} | {:<28} | {:>4.1} GB | {}",
            status, meta.id, meta.category, meta.size_gb, meta.speed_m2_pro
        );
        println!("       ↳ {}", meta.description);
    }
    println!("   ─────────────────────────────────────────────────────────────────────────────");
    println!("   Download with: nirvana-code download <id>\n");
    Ok(())
}

async fn cmd_benchmark(
    cli: &Cli,
    num_tokens: usize,
    runs: usize,
    json: bool,
    prompt_tokens: usize,
) -> Result<()> {
    let model_path = ModelManager::resolve_model_path(cli.model.as_deref())
        .context("No model found for benchmark. Run 'nirvana-code download qwen-1.5b'")?;
    let kv_mode = cli.kv_mode();
    let hw = hardware::SiliconProfile::detect();
    let runs = runs.max(1);

    if !json {
        println!("\n⚡ Nirvana Code benchmark");
        println!(
            "   Chip:        {} ({}P+{}E, {} GPU cores, {} GB)",
            hw.chip_name, hw.p_cores, hw.e_cores, hw.gpu_cores, hw.memory_gb
        );
        println!("   Model:       {}", model_path.display());
        println!("   KV-Cache:    {}", kv_mode.label());
        println!(
            "   Context:     {} tokens · prompt ≈{} tokens · {} output tokens · {} runs",
            cli.ctx_size, prompt_tokens, num_tokens, runs
        );
        println!();
    }

    let engine = load_engine(cli, &model_path, kv_mode)?;

    // A prompt of roughly `prompt_tokens` tokens so prefill is measurable
    let filler = "fn compute(x: i32) -> i32 { x * 2 + 1 } ";
    let mut body = String::new();
    while body.len() < prompt_tokens * 3 {
        body.push_str(filler);
    }
    let system = "You are Nirvana Code, an Apple Silicon coding assistant. Answer with code.";
    let cold_prompt = vec![
        chat::ChatMessage::system(system),
        chat::ChatMessage::user(format!(
            "Here is some code:\n{body}\nWrite a fast concurrent queue in Rust."
        )),
    ];
    let warm_prompt = vec![
        chat::ChatMessage::system(system),
        chat::ChatMessage::user(format!(
            "Here is some code:\n{body}\nExplain how the queue avoids data races."
        )),
    ];

    let config = GenerationConfig {
        max_tokens: num_tokens,
        temperature: 0.0,
        use_ngram_speculative: cli.ngram_speculative,
        seed: Some(1),
        ..GenerationConfig::default()
    };

    #[derive(Default, Clone, Copy)]
    struct Sample {
        ttft_ms: f64,
        prefill_tps: f64,
        decode_tps: f64,
        prompt_tokens: usize,
        prefix_reused: usize,
    }

    async fn run_once(
        engine: &InferenceEngine,
        msgs: &[chat::ChatMessage],
        cfg: &GenerationConfig,
    ) -> Result<Sample> {
        let (tx, mut rx) = unbounded_channel();
        let eng = engine.clone();
        let msgs = msgs.to_vec();
        let cfg = cfg.clone();
        tokio::task::spawn_blocking(move || {
            let tx_err = tx.clone();
            if let Err(e) = eng.stream_chat(&msgs, &cfg, Arc::new(AtomicBool::new(false)), tx) {
                let _ = tx_err.send(StreamEvent::Error(e.to_string()));
            }
        });
        let mut out = Sample::default();
        while let Some(ev) = rx.recv().await {
            match ev {
                StreamEvent::Stats {
                    ttft_ms,
                    tokens_per_sec,
                    prompt_tokens,
                    prefix_tokens_reused,
                    ..
                } => {
                    out.ttft_ms = ttft_ms as f64;
                    out.decode_tps = tokens_per_sec;
                    out.prompt_tokens = prompt_tokens;
                    out.prefix_reused = prefix_tokens_reused;
                    let evaluated = prompt_tokens.saturating_sub(prefix_tokens_reused);
                    out.prefill_tps = if ttft_ms > 0 {
                        evaluated as f64 / (ttft_ms as f64 / 1000.0)
                    } else {
                        0.0
                    };
                }
                StreamEvent::Error(e) => anyhow::bail!("benchmark generation failed: {e}"),
                _ => {}
            }
        }
        Ok(out)
    }

    fn median(v: &mut [f64]) -> f64 {
        if v.is_empty() {
            return 0.0;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = v.len();
        if n % 2 == 1 {
            v[n / 2]
        } else {
            (v[n / 2 - 1] + v[n / 2]) / 2.0
        }
    }

    // Warm-up pass: Metal shader compilation and first-touch page faults
    engine.clear_cache();
    let _ = run_once(
        &engine,
        &cold_prompt,
        &GenerationConfig {
            max_tokens: 4,
            ..config.clone()
        },
    )
    .await?;

    let mut cold: Vec<Sample> = Vec::with_capacity(runs);
    let mut warm: Vec<Sample> = Vec::with_capacity(runs);
    for i in 0..runs {
        engine.clear_cache();
        let c = run_once(&engine, &cold_prompt, &config).await?;
        let w = run_once(&engine, &warm_prompt, &config).await?;
        if !json {
            println!(
                "   run {:>2}: cold TTFT {:>6.0} ms ({:>6.0} tok/s prefill) · warm TTFT {:>5.0} ms ({} reused) · decode {:>6.1} tok/s",
                i + 1,
                c.ttft_ms,
                c.prefill_tps,
                w.ttft_ms,
                w.prefix_reused,
                c.decode_tps
            );
        }
        cold.push(c);
        warm.push(w);
    }

    let m =
        |f: &dyn Fn(&Sample) -> f64, v: &[Sample]| median(&mut v.iter().map(f).collect::<Vec<_>>());
    let cold_ttft = m(&|s| s.ttft_ms, &cold);
    let warm_ttft = m(&|s| s.ttft_ms, &warm);
    let prefill = m(&|s| s.prefill_tps, &cold);
    let decode = m(&|s| s.decode_tps, &cold);
    let decode_warm = m(&|s| s.decode_tps, &warm);
    let reused = warm.first().map(|s| s.prefix_reused).unwrap_or(0);
    let n_prompt = cold.first().map(|s| s.prompt_tokens).unwrap_or(0);

    if json {
        let out = serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "chip": hw.chip_name,
            "p_cores": hw.p_cores, "e_cores": hw.e_cores, "gpu_cores": hw.gpu_cores, "memory_gb": hw.memory_gb,
            "model": model_path.file_name().map(|n| n.to_string_lossy().to_string()),
            "backend": engine.backend_name(),
            "kv_cache": engine.kv_label(),
            "ctx": cli.ctx_size,
            "ubatch": engine::ubatch_size(),
            "runs": runs,
            "prompt_tokens": n_prompt,
            "output_tokens": num_tokens,
            "median": {
                "cold_ttft_ms": cold_ttft,
                "warm_ttft_ms": warm_ttft,
                "prefill_tok_s": prefill,
                "decode_tok_s": decode,
                "decode_tok_s_warm": decode_warm,
                "prefix_tokens_reused": reused,
            },
            "runs_cold_ttft_ms": cold.iter().map(|s| s.ttft_ms).collect::<Vec<_>>(),
            "runs_decode_tok_s": cold.iter().map(|s| s.decode_tps).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        let reduction = if cold_ttft > 0.0 {
            (cold_ttft - warm_ttft) / cold_ttft * 100.0
        } else {
            0.0
        };
        println!("\n   median of {runs} runs");
        println!(
            "   prefill:     {prefill:>7.0} tok/s   (cold TTFT {cold_ttft:.0} ms over {n_prompt} prompt tokens)"
        );
        println!(
            "   warm TTFT:   {warm_ttft:>7.0} ms      ({reused} prefix tokens reused, {reduction:.0}% lower than cold)"
        );
        println!("   decode:      {decode:>7.1} tok/s   (warm run {decode_warm:.1})");
        println!(
            "   backend:     {} · {}",
            engine.backend_name(),
            engine.kv_label()
        );
        println!();
    }

    Ok(())
}

async fn cmd_single_shot(cli: &Cli, prompt: &str, preset: &str) -> Result<()> {
    let model_path = ModelManager::resolve_model_path(cli.model.as_deref())
        .context("No model found. Run 'nirvana-code download qwen-1.5b'")?;

    let tmpl = TEMPLATES
        .iter()
        .find(|t| t.id == preset)
        .unwrap_or(&TEMPLATES[0]);

    let messages = tmpl.messages(prompt);

    let kv_mode = cli.kv_mode();

    let engine = load_engine(cli, &model_path, kv_mode)?;

    let (tx, mut rx) = unbounded_channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let config = GenerationConfig {
        max_tokens: cli.max_tokens,
        temperature: cli.temperature,
        min_p: cli.min_p,
        top_p: cli.top_p,
        top_k: cli.top_k,
        use_ngram_speculative: cli.ngram_speculative,
        seed: cli.seed,
        repeat_penalty: cli.repeat_penalty,
        dry_multiplier: cli.dry_multiplier,
        ..GenerationConfig::default()
    };

    let engine_after = engine.clone();
    tokio::task::spawn_blocking(move || {
        let tx_err = tx.clone();
        if let Err(e) = engine.stream_chat(&messages, &config, cancel, tx) {
            let _ = tx_err.send(StreamEvent::Error(e.to_string()));
        }
    });

    use std::io::Write;
    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::Token(tok) => {
                print!("{tok}");
                let _ = io::stdout().flush();
            }
            StreamEvent::Stats {
                ttft_ms,
                tokens_per_sec,
                total_tokens,
                prefix_cache_hit,
                kv_type,
                ..
            } => {
                println!(
                    "\n\n[Stats: TTFT: {ttft_ms}ms | {tokens_per_sec:.1} tok/s | {total_tokens} tokens | Prefix Hit: {prefix_cache_hit} | {kv_type}]"
                );
            }
            StreamEvent::Done => break,
            StreamEvent::Error(err) => {
                eprintln!("\nError: {err}");
                break;
            }
        }
    }

    persist_session(cli, &engine_after);
    Ok(())
}

async fn cmd_serve(cli: &Cli, port: u16, host: &str, socket: Option<&Path>) -> Result<()> {
    let model_path = match ModelManager::resolve_model_path(cli.model.as_deref()) {
        Some(p) => p,
        None => {
            eprintln!("\n❌ No GGUF model found!");
            eprintln!("Run: nirvana-code download qwen-1.5b\n");
            return Ok(());
        }
    };

    let kv_mode = cli.kv_mode();

    let engine = load_engine(cli, &model_path, kv_mode)?;

    let model_name = model_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "nirvana-code".to_string());

    let opts = server_options(
        cli,
        host,
        port,
        socket.map(|p| p.to_path_buf()),
        kv_mode,
        false,
    )?;
    server::run_server(engine, model_name, model_path, opts).await?;
    Ok(())
}

async fn cmd_web(cli: &Cli, port: u16, host: &str, open_browser: bool) -> Result<()> {
    let model_path = match ModelManager::resolve_model_path(cli.model.as_deref()) {
        Some(p) => p,
        None => {
            eprintln!("\n❌ No GGUF model found!");
            eprintln!("Run: nirvana-code download qwen-1.5b\n");
            return Ok(());
        }
    };

    let kv_mode = cli.kv_mode();

    let engine = load_engine(cli, &model_path, kv_mode)?;

    let model_name = model_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "nirvana-code".to_string());

    if open_browser {
        let host_clone = host.to_string();
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(600)).await;
            #[cfg(target_os = "macos")]
            let _ = std::process::Command::new("open")
                .arg(format!("http://{host_clone}:{port}"))
                .spawn();
            #[cfg(target_os = "linux")]
            let _ = std::process::Command::new("xdg-open")
                .arg(format!("http://{}:{}", host_clone, port))
                .spawn();
            #[cfg(target_os = "windows")]
            let _ = std::process::Command::new("cmd")
                .args(["/C", "start", &format!("http://{}:{}", host_clone, port)])
                .spawn();
        });
    }

    let opts = server_options(cli, host, port, None, kv_mode, true)?;
    server::run_server(engine, model_name, model_path, opts).await?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_tui(
    engine: InferenceEngine,
    model_path: PathBuf,
    max_tokens: usize,
    temperature: f32,
    min_p: f32,
    top_p: f32,
    top_k: i32,
    ngram_speculative: bool,
    seed: Option<u32>,
    persist_kv: bool,
) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(
        engine,
        model_path,
        max_tokens,
        temperature,
        min_p,
        top_p,
        top_k,
        ngram_speculative,
        seed,
    );

    let last_tick = Instant::now();
    let tick_rate = Duration::from_millis(16); // 60 FPS

    let res = run_app_loop(&mut terminal, &mut app, last_tick, tick_rate);

    // Restore terminal cleanly
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(e) = res {
        eprintln!("Nirvana Code execution error: {e:?}");
    }

    if persist_kv {
        app.cancel_generation();
        match app.engine.save_session() {
            Ok(n) if n > 0 => eprintln!("KV state: saved {n} prefix tokens"),
            Ok(_) => {}
            Err(e) => eprintln!("KV state: save failed ({e})"),
        }
    }

    Ok(())
}

fn run_app_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    mut last_tick: Instant,
    tick_rate: Duration,
) -> Result<()> {
    loop {
        app.poll_stream();
        app.update_toast();

        terminal.draw(|f| ui::draw(f, app))?;

        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)? {
            match event::read()? {
                Event::Mouse(mouse) => match mouse.kind {
                    MouseEventKind::ScrollUp => {
                        app.auto_scroll = false;
                        app.scroll_offset = app.scroll_offset.saturating_sub(3);
                    }
                    MouseEventKind::ScrollDown => {
                        app.scroll_offset = (app.scroll_offset + 3).min(app.max_scroll);
                        if app.scroll_offset >= app.max_scroll {
                            app.auto_scroll = true;
                        }
                    }
                    _ => {}
                },
                Event::Key(key) => {
                    if key.kind == event::KeyEventKind::Press {
                        // 1. Exit Confirmation Modal intercept
                        if app.exit_confirmation {
                            match key.code {
                                KeyCode::Char('c')
                                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                                {
                                    return Ok(());
                                }
                                KeyCode::Char('y') | KeyCode::Char('Y') => {
                                    return Ok(());
                                }
                                _ => {
                                    app.exit_confirmation = false;
                                    app.exit_confirmation_time = None;
                                    app.set_toast("Exit cancelled");
                                    continue;
                                }
                            }
                        }

                        // 1b. Model Picker Modal intercept
                        if app.show_model_picker {
                            match key.code {
                                KeyCode::Esc => {
                                    app.show_model_picker = false;
                                }
                                KeyCode::Char('c')
                                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                                {
                                    app.show_model_picker = false;
                                }
                                KeyCode::Up => {
                                    if app.model_picker_index > 0 {
                                        app.model_picker_index -= 1;
                                    }
                                }
                                KeyCode::Down => {
                                    if !app.installed_models.is_empty()
                                        && app.model_picker_index + 1 < app.installed_models.len()
                                    {
                                        app.model_picker_index += 1;
                                    }
                                }
                                KeyCode::Enter => {
                                    if let Some((path, _, _)) =
                                        app.installed_models.get(app.model_picker_index)
                                    {
                                        let p = path.clone();
                                        let _ = app.switch_model(p);
                                    }
                                    app.show_model_picker = false;
                                }
                                KeyCode::Char(c) => {
                                    if let Some(digit) = c.to_digit(10) {
                                        let idx = digit as usize;
                                        if idx >= 1 && idx <= app.installed_models.len() {
                                            let path = app.installed_models[idx - 1].0.clone();
                                            let _ = app.switch_model(path);
                                            app.show_model_picker = false;
                                        }
                                    }
                                }
                                _ => {}
                            }
                            continue;
                        }

                        // 1c. Attach File Modal intercept
                        if app.show_attach_modal {
                            match key.code {
                                KeyCode::Esc => {
                                    app.close_attach_modal();
                                }
                                KeyCode::Char('c')
                                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                                {
                                    app.close_attach_modal();
                                }
                                KeyCode::Enter => {
                                    let path_to_attach = app.attach_input.trim().to_string();
                                    if !path_to_attach.is_empty() {
                                        if let Err(e) = app.attach_file(&path_to_attach) {
                                            app.set_toast(&format!("❌ Failed to attach: {e}"));
                                        }
                                    }
                                    app.close_attach_modal();
                                }
                                KeyCode::Backspace => {
                                    app.attach_input.pop();
                                }
                                KeyCode::Char(c) => {
                                    app.attach_input.push(c);
                                }
                                _ => {}
                            }
                            continue;
                        }

                        // 2. Command Palette Interaction
                        if app.show_palette {
                            match key.code {
                                KeyCode::Esc => {
                                    app.show_palette = false;
                                }
                                KeyCode::Char('c')
                                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                                {
                                    app.show_palette = false;
                                }
                                KeyCode::Up => {
                                    if app.palette_index > 0 {
                                        app.palette_index -= 1;
                                    }
                                }
                                KeyCode::Down => {
                                    let filtered = PaletteManager::filter_items(
                                        &app.palette_items,
                                        &app.palette_query,
                                    );
                                    if !filtered.is_empty()
                                        && app.palette_index + 1 < filtered.len()
                                    {
                                        app.palette_index += 1;
                                    }
                                }
                                KeyCode::Enter => {
                                    let filtered = PaletteManager::filter_items(
                                        &app.palette_items,
                                        &app.palette_query,
                                    );
                                    if let Some(item) = filtered.get(app.palette_index) {
                                        let action = item.action.clone();
                                        app.execute_palette_action(action);
                                    }
                                }
                                KeyCode::Backspace => {
                                    app.palette_query.pop();
                                    app.palette_index = 0;
                                }
                                KeyCode::Char(c) => {
                                    app.palette_query.push(c);
                                    app.palette_index = 0;
                                }
                                _ => {}
                            }
                            continue;
                        }

                        // 3. Global Shortcuts
                        if key.modifiers.contains(KeyModifiers::CONTROL) {
                            match key.code {
                                KeyCode::Char('k') => {
                                    app.show_palette = !app.show_palette;
                                    app.palette_query.clear();
                                    app.palette_index = 0;
                                    continue;
                                }
                                KeyCode::Char('c') => {
                                    if app.engine_state == EngineState::Generating {
                                        app.cancel_generation();
                                    } else {
                                        app.exit_confirmation = true;
                                        app.exit_confirmation_time = Some(Instant::now());
                                        app.set_toast("⚠️ Press Ctrl+C again to confirm exit");
                                    }
                                    continue;
                                }
                                KeyCode::Char('b') => {
                                    app.show_sidebar = !app.show_sidebar;
                                    app.set_toast(if app.show_sidebar {
                                        "✔ Sidebar visible"
                                    } else {
                                        "✔ Sidebar hidden (Full Workspace)"
                                    });
                                    continue;
                                }
                                KeyCode::Char('p') => {
                                    app.open_model_picker();
                                    continue;
                                }
                                KeyCode::Char('f') => {
                                    app.open_attach_modal();
                                    continue;
                                }
                                KeyCode::Char('s') => {
                                    app.send_input();
                                    continue;
                                }
                                KeyCode::Char('o') => {
                                    app.copy_last_response();
                                    continue;
                                }
                                KeyCode::Char('y') => {
                                    app.copy_first_code_snippet();
                                    continue;
                                }
                                KeyCode::Char('r') => {
                                    app.chat_history.clear();
                                    app.current_stream.clear();
                                    app.engine.clear_cache();
                                    app.set_toast("✔ KV Cache & conversation history cleared");
                                    continue;
                                }
                                KeyCode::Char('u') => {
                                    app.auto_scroll = false;
                                    app.scroll_offset = app.scroll_offset.saturating_sub(10);
                                    continue;
                                }
                                KeyCode::Char('d') => {
                                    if app.current_attachment.is_some() {
                                        app.detach_file();
                                    } else {
                                        app.scroll_offset =
                                            (app.scroll_offset + 10).min(app.max_scroll);
                                        if app.scroll_offset >= app.max_scroll {
                                            app.auto_scroll = true;
                                        }
                                    }
                                    continue;
                                }
                                KeyCode::Up => {
                                    app.auto_scroll = false;
                                    app.scroll_offset = app.scroll_offset.saturating_sub(3);
                                    continue;
                                }
                                KeyCode::Down => {
                                    app.scroll_offset = (app.scroll_offset + 3).min(app.max_scroll);
                                    if app.scroll_offset >= app.max_scroll {
                                        app.auto_scroll = true;
                                    }
                                    continue;
                                }
                                _ => {}
                            }
                        }

                        // 4. Shift & Alt scrolling shortcuts
                        if key.modifiers.contains(KeyModifiers::SHIFT)
                            || key.modifiers.contains(KeyModifiers::ALT)
                        {
                            match key.code {
                                KeyCode::Up => {
                                    app.auto_scroll = false;
                                    app.scroll_offset = app.scroll_offset.saturating_sub(3);
                                    continue;
                                }
                                KeyCode::Down => {
                                    app.scroll_offset = (app.scroll_offset + 3).min(app.max_scroll);
                                    if app.scroll_offset >= app.max_scroll {
                                        app.auto_scroll = true;
                                    }
                                    continue;
                                }
                                KeyCode::Enter => {
                                    app.input_textarea.insert_newline();
                                    continue;
                                }
                                _ => {}
                            }
                        }

                        // 5. Normal Input & Workspace Scrolling Handling
                        let is_input_empty = app.input_textarea.lines().join("").trim().is_empty();
                        let cursor_at_top = app.input_textarea.cursor().0 == 0;
                        let num_lines = app.input_textarea.lines().len();

                        match key.code {
                            KeyCode::Esc => {
                                if app.current_attachment.is_some() {
                                    app.detach_file();
                                } else if is_input_empty {
                                    app.exit_confirmation = true;
                                    app.exit_confirmation_time = Some(Instant::now());
                                    app.set_toast("⚠️ Press Ctrl+C or Y to confirm exit");
                                } else {
                                    app.input_textarea = tui_textarea::TextArea::default();
                                    app.input_textarea.set_placeholder_text("Type your prompt or code question... (Enter to send, Shift+Enter for newline, Ctrl+K for palette)");
                                }
                            }
                            KeyCode::PageUp => {
                                app.auto_scroll = false;
                                app.scroll_offset = app.scroll_offset.saturating_sub(6);
                            }
                            KeyCode::PageDown => {
                                app.scroll_offset = (app.scroll_offset + 6).min(app.max_scroll);
                                if app.scroll_offset >= app.max_scroll {
                                    app.auto_scroll = true;
                                }
                            }
                            KeyCode::Up => {
                                if is_input_empty || num_lines <= 1 || cursor_at_top {
                                    app.auto_scroll = false;
                                    app.scroll_offset = app.scroll_offset.saturating_sub(3);
                                } else {
                                    app.input_textarea.input(key);
                                }
                            }
                            KeyCode::Down => {
                                if (!app.auto_scroll || is_input_empty)
                                    && app.scroll_offset < app.max_scroll
                                {
                                    app.scroll_offset = (app.scroll_offset + 3).min(app.max_scroll);
                                    if app.scroll_offset >= app.max_scroll {
                                        app.auto_scroll = true;
                                    }
                                } else {
                                    app.input_textarea.input(key);
                                }
                            }
                            KeyCode::Home => {
                                if is_input_empty {
                                    app.auto_scroll = false;
                                    app.scroll_offset = 0;
                                } else {
                                    app.input_textarea.input(key);
                                }
                            }
                            KeyCode::End => {
                                if is_input_empty {
                                    app.auto_scroll = true;
                                    app.scroll_offset = app.max_scroll;
                                } else {
                                    app.input_textarea.input(key);
                                }
                            }
                            KeyCode::Enter => {
                                app.send_input();
                            }
                            _ => {
                                app.input_textarea.input(key);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }
}
