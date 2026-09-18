mod app;
mod cli;
mod clipboard;
mod engine;
mod hardware;
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
use clap::Parser;
use cli::{Cli, Commands};
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use engine::{GenerationConfig, KvQuantMode, ModelEngine, StreamEvent};
use model_manager::{ModelManager, MODEL_CATALOG};
use palette::PaletteManager;
use ratatui::{backend::CrosstermBackend, Terminal};
use speculative::SpeculativeEngine;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};
use templates::TEMPLATES;
use tokio::sync::mpsc::unbounded_channel;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Models) => {
            cmd_list_models()?;
            return Ok(());
        }
        Some(Commands::Download { target }) => {
            ModelManager::download_target(target).await?;
            return Ok(());
        }
        Some(Commands::Bench { num_tokens }) => {
            cmd_benchmark(&cli, *num_tokens).await?;
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
        _ => {}
    }

    // Resolve model path
    let model_path = match ModelManager::resolve_model_path(cli.model.as_deref()) {
        Some(p) => p,
        None => {
            eprintln!("\n❌ No GGUF model found!");
            eprintln!("Run the following command to download the recommended model:");
            eprintln!("   nirvana-code download qwen-1.5b\n");
            eprintln!("Or download a 16GB flagship model from the AtomicChat guide:");
            eprintln!("   nirvana-code download qwen-3.8-27b");
            eprintln!("   nirvana-code download qwen-3.5-9b\n");
            return Ok(());
        }
    };

    let kv_mode = match cli.kv_type.to_lowercase().as_str() {
        "q4_0" => KvQuantMode::Q4_0,
        "f16" => KvQuantMode::F16,
        "q8_0" => KvQuantMode::Q8_0,
        _ => KvQuantMode::Auto,
    };

    let use_mlock = !cli.no_mlock;

    println!("⚡ Loading Nirvana Code Silicon Engine...");
    println!("   Model:       {}", model_path.display());
    println!("   KV-Cache:    {}", kv_mode.label());
    println!("   MLock:       {}", if use_mlock { "Enabled (LPDDR5 RAM Pinned)" } else { "Disabled" });
    println!("   Metal GPU:   {} layers offloaded", cli.gpu_layers);
    println!("   Context:     {} tokens", cli.ctx_size);
    if cli.ngram_speculative {
        println!("   Speculation: Prompt Lookup Decoding (N-gram matching) ENABLED");
    }
    println!();

    let engine = Arc::new(ModelEngine::load(
        &model_path,
        cli.gpu_layers,
        use_mlock,
        kv_mode,
        cli.ctx_size,
    )?);

    run_tui(
        engine,
        model_path,
        cli.max_tokens,
        cli.temperature,
        cli.min_p,
        cli.top_p,
        cli.top_k,
        cli.ngram_speculative,
    )?;
    Ok(())
}

fn cmd_list_models() -> Result<()> {
    println!("\n⚡ [NIRVANA CODE] Silicon Model Manager (Apple M2 Pro 16GB Unified RAM)\n");

    let installed = ModelManager::list_installed();
    println!("📦 Installed Local Models (Checked ~/.nirvana/models & ~/.promptcraft/models):");
    if installed.is_empty() {
        println!("   (None installed yet. Run 'nirvana-code download qwen-1.5b' to download)");
    } else {
        for (path, name, size) in &installed {
            let size_mb = size / (1024 * 1024);
            println!("   ✔ {:<42} {:>6} MB   {}", name, size_mb, path.display());
        }
    }

    println!("\n🌐 AtomicChat 16GB Hardware Guide Models (Ready to Download):");
    println!("   ─────────────────────────────────────────────────────────────────────────────");
    for meta in MODEL_CATALOG {
        let is_inst = installed.iter().any(|(_, name, _)| name == meta.filename);
        let status = if is_inst { "✔ INSTALLED" } else { "  AVAILABLE" };
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

async fn cmd_benchmark(cli: &Cli, num_tokens: usize) -> Result<()> {
    let model_path = ModelManager::resolve_model_path(cli.model.as_deref())
        .context("No model found for benchmark. Run 'nirvana-code download qwen-1.5b'")?;

    let kv_mode = match cli.kv_type.to_lowercase().as_str() {
        "q4_0" => KvQuantMode::Q4_0,
        "f16" => KvQuantMode::F16,
        "q8_0" => KvQuantMode::Q8_0,
        _ => KvQuantMode::Auto,
    };

    println!("\n⚡ Running Silicon Core Benchmark on Apple Silicon...");
    println!("   Model:       {}", model_path.display());
    println!("   KV-Cache:    {}", kv_mode.label());
    println!("   Memory:      mlock pinned in Unified RAM");
    if cli.ngram_speculative {
        println!("   Speculation: Prompt Lookup Decoding (N-gram matching) ENABLED");
    }
    println!();

    let engine = Arc::new(ModelEngine::load(&model_path, cli.gpu_layers, true, kv_mode, cli.ctx_size)?);

    let system_prompt = "You are Nirvana Code, an ultra-fast Apple Silicon coding assistant. Provide clean Rust code.";
    let test_prompt_1 = format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\nWrite a fast concurrent queue in Rust using atomic pointers.<|im_end|>\n<|im_start|>assistant\n",
        system_prompt
    );

    let test_prompt_2 = format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\nExplain how the concurrent queue prevents data races.<|im_end|>\n<|im_start|>assistant\n",
        system_prompt
    );

    let config = GenerationConfig {
        max_tokens: num_tokens,
        temperature: 0.0,
        min_p: cli.min_p,
        top_p: cli.top_p,
        top_k: cli.top_k,
        use_ngram_speculative: cli.ngram_speculative,
    };

    // Turn 1: Cold Cache Prefill
    println!("🚀 [Turn 1] Cold Cache Prefill (Evaluating system + user prompt from scratch)...");
    let (tx1, mut rx1) = unbounded_channel();
    let cancel1 = Arc::new(AtomicBool::new(false));

    let eng1 = engine.clone();
    let cfg1 = config.clone();
    tokio::task::spawn_blocking(move || {
        let _ = eng1.stream_generate_with_config(&test_prompt_1, &cfg1, cancel1, tx1);
    });

    let mut ttft_cold = 0;
    let mut tps_cold = 0.0;
    while let Some(ev) = rx1.recv().await {
        if let StreamEvent::Stats { ttft_ms, tokens_per_sec, .. } = ev {
            ttft_cold = ttft_ms;
            tps_cold = tokens_per_sec;
        }
    }
    println!("   Cold TTFT:    {} ms", ttft_cold);
    println!("   Decode Speed: {:.1} tokens/sec\n", tps_cold);

    // Turn 2: Warm Prefix Cache Reuse
    println!("⚡ [Turn 2] Warm Prefix Cache Reuse (Reusing system prompt KV state)...");
    let (tx2, mut rx2) = unbounded_channel();
    let cancel2 = Arc::new(AtomicBool::new(false));

    let eng2 = engine.clone();
    let cfg2 = config.clone();
    tokio::task::spawn_blocking(move || {
        let _ = eng2.stream_generate_with_config(&test_prompt_2, &cfg2, cancel2, tx2);
    });

    let mut ttft_warm = 0;
    let mut tps_warm = 0.0;
    let mut prefix_reused = 0;
    while let Some(ev) = rx2.recv().await {
        if let StreamEvent::Stats { ttft_ms, tokens_per_sec, prefix_tokens_reused, .. } = ev {
            ttft_warm = ttft_ms;
            tps_warm = tokens_per_sec;
            prefix_reused = prefix_tokens_reused;
        }
    }
    println!("   Warm TTFT:    {} ms (Prefix tokens reused: {})", ttft_warm, prefix_reused);
    println!("   Decode Speed: {:.1} tokens/sec\n", tps_warm);

    let speedup = if ttft_cold > 0 && ttft_warm < ttft_cold {
        ((ttft_cold as f64 - ttft_warm as f64) / ttft_cold as f64) * 100.0
    } else {
        0.0
    };

    println!("🏆 BENCHMARK RESULTS:");
    println!("   Prefix Caching TTFT Reduction: {:.1}% latency reduction!", speedup);
    println!("   KV-Cache Quantization:         {} active.", engine.kv_mode.label());
    println!("   Memory Locking (mlock):        Zero virtual memory page faults.\n");

    Ok(())
}

async fn cmd_single_shot(cli: &Cli, prompt: &str, preset: &str) -> Result<()> {
    let model_path = ModelManager::resolve_model_path(cli.model.as_deref())
        .context("No model found. Run 'nirvana-code download qwen-1.5b'")?;

    let tmpl = TEMPLATES
        .iter()
        .find(|t| t.id == preset)
        .unwrap_or(&TEMPLATES[0]);

    let full_prompt = tmpl.build_full_context(prompt);

    // If draft model is supplied or speculative is requested
    if cli.speculative || cli.draft_model.is_some() {
        let draft_path = match ModelManager::resolve_model_path(cli.draft_model.as_deref()) {
            Some(p) => p,
            None => {
                ModelManager::resolve_model_path(Some(Path::new("qwen-0.5b")))
                    .or_else(|| ModelManager::resolve_model_path(Some(Path::new("qwen-1.5b"))))
                    .context("No draft model found for speculative decoding. Run 'nirvana-code download qwen-0.5b'")?
            }
        };

        println!("⚡ [SPECULATIVE DECODING] Dual-Engine Metal Generation");
        println!("   Target Model: {}", model_path.display());
        println!("   Draft Model:  {}\n", draft_path.display());

        let engine = SpeculativeEngine::load(
            &model_path,
            &draft_path,
            cli.gpu_layers,
            !cli.no_mlock,
            cli.ctx_size,
            4,
        )?;

        let (tx, mut rx) = unbounded_channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let max_tokens = cli.max_tokens;
        let temperature = cli.temperature;

        tokio::task::spawn_blocking(move || {
            let _ = engine.stream_generate(&full_prompt, max_tokens, temperature, cancel, tx);
        });

        use std::io::Write;
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(tok) => {
                    print!("{}", tok);
                    let _ = io::stdout().flush();
                }
                StreamEvent::Stats {
                    ttft_ms,
                    tokens_per_sec,
                    total_tokens,
                    kv_type,
                    ..
                } => {
                    println!(
                        "\n\n[Stats: TTFT: {}ms | {:.1} tok/s | {} tokens | {}]",
                        ttft_ms, tokens_per_sec, total_tokens, kv_type
                    );
                }
                StreamEvent::Done => break,
                StreamEvent::Error(err) => {
                    eprintln!("\nError: {}", err);
                    break;
                }
            }
        }
        return Ok(());
    }

    let kv_mode = match cli.kv_type.to_lowercase().as_str() {
        "q4_0" => KvQuantMode::Q4_0,
        "f16" => KvQuantMode::F16,
        "q8_0" => KvQuantMode::Q8_0,
        _ => KvQuantMode::Auto,
    };

    let engine = ModelEngine::load(&model_path, cli.gpu_layers, !cli.no_mlock, kv_mode, cli.ctx_size)?;

    let (tx, mut rx) = unbounded_channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let config = GenerationConfig {
        max_tokens: cli.max_tokens,
        temperature: cli.temperature,
        min_p: cli.min_p,
        top_p: cli.top_p,
        top_k: cli.top_k,
        use_ngram_speculative: cli.ngram_speculative,
    };

    tokio::task::spawn_blocking(move || {
        let _ = engine.stream_generate_with_config(&full_prompt, &config, cancel, tx);
    });

    use std::io::Write;
    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::Token(tok) => {
                print!("{}", tok);
                let _ = io::stdout().flush();
            }
            StreamEvent::Stats {
                ttft_ms,
                tokens_per_sec,
                total_tokens,
                prefix_cache_hit,
                ..
            } => {
                println!(
                    "\n\n[Stats: TTFT: {}ms | {:.1} tok/s | {} tokens | Prefix Hit: {}]",
                    ttft_ms, tokens_per_sec, total_tokens, prefix_cache_hit
                );
            }
            StreamEvent::Done => break,
            StreamEvent::Error(err) => {
                eprintln!("\nError: {}", err);
                break;
            }
        }
    }

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

    let kv_mode = match cli.kv_type.to_lowercase().as_str() {
        "q4_0" => KvQuantMode::Q4_0,
        "f16" => KvQuantMode::F16,
        "q8_0" => KvQuantMode::Q8_0,
        _ => KvQuantMode::Auto,
    };

    let engine = Arc::new(ModelEngine::load(
        &model_path,
        cli.gpu_layers,
        !cli.no_mlock,
        kv_mode,
        cli.ctx_size,
    )?);

    let model_name = model_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "nirvana-code".to_string());

    server::run_server(engine, model_name, host, port, socket.map(|p| p.to_path_buf())).await?;
    Ok(())
}

fn run_tui(
    engine: Arc<ModelEngine>,
    model_path: PathBuf,
    max_tokens: usize,
    temperature: f32,
    min_p: f32,
    top_p: f32,
    top_k: i32,
    ngram_speculative: bool,
) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
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
    );

    let last_tick = Instant::now();
    let tick_rate = Duration::from_millis(16); // 60 FPS

    let res = run_app_loop(&mut terminal, &mut app, last_tick, tick_rate);

    // Restore terminal cleanly
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(e) = res {
        eprintln!("Nirvana Code execution error: {:?}", e);
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
            if let Event::Key(key) = event::read()? {
                if key.kind == event::KeyEventKind::Press {
                    // Global Shortcuts
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
                                    return Ok(());
                                }
                                continue;
                            }
                            KeyCode::Char('s') => {
                                app.send_input();
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
                            _ => {}
                        }
                    }

                    // Command Palette Interaction
                    if app.show_palette {
                        match key.code {
                            KeyCode::Esc => {
                                app.show_palette = false;
                            }
                            KeyCode::Up => {
                                if app.palette_index > 0 {
                                    app.palette_index -= 1;
                                }
                            }
                            KeyCode::Down => {
                                let filtered =
                                    PaletteManager::filter_items(&app.palette_items, &app.palette_query);
                                if !filtered.is_empty() && app.palette_index + 1 < filtered.len() {
                                    app.palette_index += 1;
                                }
                            }
                            KeyCode::Enter => {
                                let filtered =
                                    PaletteManager::filter_items(&app.palette_items, &app.palette_query);
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

                    // Normal Input Handling
                    match key.code {
                        KeyCode::Esc => {
                            let text = app.input_textarea.lines().join("");
                            if text.trim().is_empty() {
                                return Ok(());
                            } else {
                                app.input_textarea = tui_textarea::TextArea::default();
                            }
                        }
                        KeyCode::PageUp => {
                            app.scroll_offset = app.scroll_offset.saturating_add(5);
                        }
                        KeyCode::PageDown => {
                            app.scroll_offset = app.scroll_offset.saturating_sub(5);
                        }
                        KeyCode::Enter => {
                            if key.modifiers.contains(KeyModifiers::CONTROL) {
                                app.send_input();
                            } else {
                                app.input_textarea.input(key);
                            }
                        }
                        _ => {
                            app.input_textarea.input(key);
                        }
                    }
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }
}
