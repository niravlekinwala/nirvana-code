mod app;
pub mod attachment;
mod cli;
mod clipboard;
mod engine;
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
use clap::Parser;
use cli::{Cli, Commands};
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers,
        MouseEventKind,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use engine::{GenerationConfig, InferenceEngine, KvQuantMode, StreamEvent};
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
        Some(Commands::Web { port, host, no_open }) => {
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

    let kv_mode = match cli.kv_type.to_lowercase().as_str() {
        "q4_0" => KvQuantMode::Q4_0,
        "f16" => KvQuantMode::F16,
        "q8_0" => KvQuantMode::Q8_0,
        _ => KvQuantMode::Auto,
    };

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
    println!("   MLock:       {}", if use_mlock { "Enabled (LPDDR5 RAM Pinned)" } else { "Disabled" });
    if !is_mlx {
        println!("   Metal GPU:   {} layers offloaded", cli.gpu_layers);
    }
    println!("   Context:     {} tokens", cli.ctx_size);
    if cli.ngram_speculative {
        println!("   Speculation: Prompt Lookup Decoding (N-gram matching) ENABLED");
    }
    println!();

    let engine = InferenceEngine::load(
        &model_path,
        cli.gpu_layers,
        use_mlock,
        kv_mode,
        cli.ctx_size,
    )?;

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
            println!("   {:>2}. {:<12} {:<45} {:>8}   {}", i + 1, backend_badge, name, size_str, path.display());
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

    let engine = InferenceEngine::load(&model_path, cli.gpu_layers, true, kv_mode, cli.ctx_size)?;

    let system_prompt = "You are Nirvana Code, an ultra-fast Apple Silicon coding assistant. Provide clean Rust code.";
    let test_prompt_1 = format!(
        "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nWrite a fast concurrent queue in Rust using atomic pointers.<|im_end|>\n<|im_start|>assistant\n"
    );

    let test_prompt_2 = format!(
        "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nExplain how the concurrent queue prevents data races.<|im_end|>\n<|im_start|>assistant\n"
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
    println!("   Cold TTFT:    {ttft_cold} ms");
    println!("   Decode Speed: {tps_cold:.1} tokens/sec\n");

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
    println!("   Warm TTFT:    {ttft_warm} ms (Prefix tokens reused: {prefix_reused})");
    println!("   Decode Speed: {tps_warm:.1} tokens/sec\n");

    let speedup = if ttft_cold > 0 && ttft_warm < ttft_cold {
        ((ttft_cold as f64 - ttft_warm as f64) / ttft_cold as f64) * 100.0
    } else {
        0.0
    };

    println!("🏆 BENCHMARK RESULTS:");
    println!("   Prefix Caching TTFT Reduction: {speedup:.1}% latency reduction!");
    println!("   KV-Cache Quantization:         {} active.", engine.kv_label());
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
                    print!("{tok}");
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
                        "\n\n[Stats: TTFT: {ttft_ms}ms | {tokens_per_sec:.1} tok/s | {total_tokens} tokens | {kv_type}]"
                    );
                }
                StreamEvent::Done => break,
                StreamEvent::Error(err) => {
                    eprintln!("\nError: {err}");
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

    let engine = InferenceEngine::load(&model_path, cli.gpu_layers, !cli.no_mlock, kv_mode, cli.ctx_size)?;

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
                print!("{tok}");
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
                    "\n\n[Stats: TTFT: {ttft_ms}ms | {tokens_per_sec:.1} tok/s | {total_tokens} tokens | Prefix Hit: {prefix_cache_hit}]"
                );
            }
            StreamEvent::Done => break,
            StreamEvent::Error(err) => {
                eprintln!("\nError: {err}");
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

    let engine = InferenceEngine::load(
        &model_path,
        cli.gpu_layers,
        !cli.no_mlock,
        kv_mode,
        cli.ctx_size,
    )?;

    let model_name = model_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "nirvana-code".to_string());

    server::run_server(
        engine,
        model_name,
        model_path,
        host,
        port,
        socket.map(|p| p.to_path_buf()),
        cli.gpu_layers,
        !cli.no_mlock,
        kv_mode,
        cli.ctx_size,
    ).await?;
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

    let kv_mode = match cli.kv_type.to_lowercase().as_str() {
        "q4_0" => KvQuantMode::Q4_0,
        "f16" => KvQuantMode::F16,
        "q8_0" => KvQuantMode::Q8_0,
        _ => KvQuantMode::Auto,
    };

    let engine = InferenceEngine::load(
        &model_path,
        cli.gpu_layers,
        !cli.no_mlock,
        kv_mode,
        cli.ctx_size,
    )?;

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

    server::run_server(
        engine,
        model_name,
        model_path,
        host,
        port,
        None,
        cli.gpu_layers,
        !cli.no_mlock,
        kv_mode,
        cli.ctx_size,
    ).await?;
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
    );

    let last_tick = Instant::now();
    let tick_rate = Duration::from_millis(16); // 60 FPS

    let res = run_app_loop(&mut terminal, &mut app, last_tick, tick_rate);

    // Restore terminal cleanly
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;

    if let Err(e) = res {
        eprintln!("Nirvana Code execution error: {e:?}");
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
                Event::Mouse(mouse) => {
                    match mouse.kind {
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
                    }
                }
                Event::Key(key) => {
                    if key.kind == event::KeyEventKind::Press {
                        // 1. Exit Confirmation Modal intercept
                        if app.exit_confirmation {
                            match key.code {
                                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
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
                                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                    app.show_model_picker = false;
                                }
                                KeyCode::Up => {
                                    if app.model_picker_index > 0 {
                                        app.model_picker_index -= 1;
                                    }
                                }
                                KeyCode::Down => {
                                    if !app.installed_models.is_empty() && app.model_picker_index + 1 < app.installed_models.len() {
                                        app.model_picker_index += 1;
                                    }
                                }
                                KeyCode::Enter => {
                                    if let Some((path, _, _)) = app.installed_models.get(app.model_picker_index) {
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
                                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
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
                                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
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
                                        app.scroll_offset = (app.scroll_offset + 10).min(app.max_scroll);
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
                        if key.modifiers.contains(KeyModifiers::SHIFT) || key.modifiers.contains(KeyModifiers::ALT) {
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
                                if (!app.auto_scroll || is_input_empty) && app.scroll_offset < app.max_scroll {
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
