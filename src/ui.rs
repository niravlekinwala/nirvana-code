use crate::app::{App, AppMode, EngineState};
use crate::model_manager::MODEL_CATALOG;
use crate::palette::PaletteManager;
use crate::syntax::SyntaxHighlighter;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, List, ListItem, Paragraph, Wrap,
};
use ratatui::Frame;

pub fn draw(f: &mut Frame, app: &mut App) {
    let size = f.area();

    // Background base
    let bg_block = Block::default().style(Style::default().bg(app.theme.bg_dark));
    f.render_widget(bg_block, size);

    // Root layout: Header (3) -> Body (Min 0) -> Status/Toast (1)
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(1),
        ])
        .split(size);

    draw_header_hud(f, app, chunks[0]);
    draw_main_body(f, app, chunks[1]);
    draw_footer_status(f, app, chunks[2]);

    // Command palette modal popup
    if app.show_palette {
        draw_command_palette(f, app, size);
    }
}

fn draw_header_hud(f: &mut Frame, app: &App, area: Rect) {
    let header_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(38),
            Constraint::Min(20),
            Constraint::Length(44),
        ])
        .split(area);

    // 1. Title & Silicon Brand
    let title_spans = vec![
        Span::styled(
            " ⚡ NIRVANA CODE ",
            Style::default()
                .fg(app.theme.neon_cyan)
                .bg(app.theme.bg_card)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "// SILICON CORE ",
            Style::default()
                .fg(app.theme.neon_magenta)
                .bg(app.theme.bg_card)
                .add_modifier(Modifier::BOLD),
        ),
    ];
    let title_p = Paragraph::new(Line::from(title_spans))
        .block(Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(app.theme.border_dim)));
    f.render_widget(title_p, header_chunks[0]);

    // 2. Silicon Badges (KV Cache Quant, mlock, Prefix Cache)
    let prefix_text = if app.current_prefix_hit {
        format!("PREFIX: HIT (+{} tok)", app.current_prefix_reused)
    } else {
        "PREFIX: READY".to_string()
    };

    let badges = vec![
        Span::styled(
            format!(" {} ", app.engine.kv_mode.label()),
            app.theme.badge_style(),
        ),
        Span::raw(" "),
        Span::styled(
            " MLOCK: LOCKED ",
            app.theme.green_badge(),
        ),
        Span::raw(" "),
        Span::styled(
            format!(" {} ", prefix_text),
            if app.current_prefix_hit {
                app.theme.green_badge()
            } else {
                app.theme.amber_badge()
            },
        ),
    ];
    let badges_p = Paragraph::new(Line::from(badges))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(app.theme.border_dim)));
    f.render_widget(badges_p, header_chunks[1]);

    // 3. Live Silicon Performance Metrics
    let ttft_str = app
        .current_ttft_ms
        .map(|ms| format!("{}ms", ms))
        .unwrap_or_else(|| "--".to_string());
    let tps_str = app
        .current_tps
        .map(|tps| format!("{:.1} tok/s", tps))
        .unwrap_or_else(|| "--".to_string());

    let perf_spans = vec![
        Span::styled("TTFT: ", Style::default().fg(app.theme.text_muted)),
        Span::styled(ttft_str, Style::default().fg(app.theme.neon_amber).add_modifier(Modifier::BOLD)),
        Span::raw("  "),
        Span::styled("Speed: ", Style::default().fg(app.theme.text_muted)),
        Span::styled(tps_str, Style::default().fg(app.theme.neon_cyan).add_modifier(Modifier::BOLD)),
        Span::raw("  "),
        Span::styled(format!("Tokens: {} ", app.current_tokens), Style::default().fg(app.theme.text_bright)),
    ];
    let perf_p = Paragraph::new(Line::from(perf_spans))
        .alignment(Alignment::Right)
        .block(Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(app.theme.border_dim)));
    f.render_widget(perf_p, header_chunks[2]);
}

fn draw_main_body(f: &mut Frame, app: &mut App, area: Rect) {
    let body_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(34), Constraint::Min(40)])
        .split(area);

    draw_sidebar(f, app, body_chunks[0]);
    draw_content_pane(f, app, body_chunks[1]);
}

fn draw_sidebar(f: &mut Frame, app: &App, area: Rect) {
    let sidebar_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),  // Mode & Template
            Constraint::Length(12), // Silicon Architecture
            Constraint::Min(8),     // AtomicChat 16GB Guide Models
        ])
        .split(area);

    // 1. Template & Mode Info
    let mut mode_lines = Vec::new();
    mode_lines.push(Line::from(vec![
        Span::styled("Mode: ", Style::default().fg(app.theme.text_muted)),
        Span::styled(
            match app.mode {
                AppMode::Chat => "Offline Coder / Chat",
                AppMode::PromptCraft => "Prompt Optimization",
                AppMode::SiliconHUD => "Silicon Telemetry",
            },
            Style::default().fg(app.theme.neon_cyan).add_modifier(Modifier::BOLD),
        ),
    ]));
    mode_lines.push(Line::from(vec![
        Span::styled("Target: ", Style::default().fg(app.theme.text_muted)),
        Span::styled(
            app.active_template.target_model,
            Style::default().fg(app.theme.neon_magenta).add_modifier(Modifier::BOLD),
        ),
    ]));
    mode_lines.push(Line::from(vec![
        Span::styled("Preset: ", Style::default().fg(app.theme.text_muted)),
        Span::styled(
            app.active_template.name,
            Style::default().fg(app.theme.text_bright),
        ),
    ]));
    mode_lines.push(Line::from(""));
    mode_lines.push(Line::from(vec![
        Span::styled("Press ", Style::default().fg(app.theme.text_muted)),
        Span::styled("Ctrl+K", Style::default().fg(app.theme.neon_amber).add_modifier(Modifier::BOLD)),
        Span::styled(" for Command Palette", Style::default().fg(app.theme.text_muted)),
    ]));

    let mode_block = Block::default()
        .title(" Active Protocol ")
        .title_style(app.theme.title_style())
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(app.theme.border_dim))
        .style(Style::default().bg(app.theme.bg_card));
    f.render_widget(Paragraph::new(mode_lines).block(mode_block), sidebar_chunks[0]);

    // 2. Hardware Specs Card
    let offloaded = app.engine.offloaded_layers();
    let total = app.engine.total_layers();
    let hw_lines = vec![
        Line::from(vec![
            Span::styled("Chip: ", Style::default().fg(app.theme.text_muted)),
            Span::styled(&app.hardware.chip_name, Style::default().fg(app.theme.text_bright).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("GPU: ", Style::default().fg(app.theme.text_muted)),
            Span::styled(
                format!("MTL0 ({} GPU cores)", app.hardware.gpu_cores),
                Style::default().fg(app.theme.neon_cyan).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Layers: ", Style::default().fg(app.theme.text_muted)),
            Span::styled(
                format!("{}/{} on Metal 3 GPU", offloaded, total),
                Style::default().fg(app.theme.neon_green).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Memory: ", Style::default().fg(app.theme.text_muted)),
            Span::styled(
                format!("{} GB ({} GB/s Unified)", app.hardware.memory_gb, app.hardware.memory_bandwidth_gbps),
                Style::default().fg(app.theme.neon_green),
            ),
        ]),
        Line::from(vec![
            Span::styled("KV Cache: ", Style::default().fg(app.theme.text_muted)),
            Span::styled("Q8_0 Quantized (Metal)", Style::default().fg(app.theme.neon_amber)),
        ]),
        Line::from(vec![
            Span::styled("Model: ", Style::default().fg(app.theme.text_muted)),
            Span::styled(&app.model_name, Style::default().fg(app.theme.text_bright)),
        ]),
    ];

    let hw_block = Block::default()
        .title(" Silicon Engine ")
        .title_style(app.theme.title_style())
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(app.theme.border_dim))
        .style(Style::default().bg(app.theme.bg_card));
    f.render_widget(Paragraph::new(hw_lines).block(hw_block), sidebar_chunks[1]);

    // 3. AtomicChat 16GB Guide Models Vault
    let mut catalog_items = Vec::new();
    for meta in MODEL_CATALOG {
        let is_installed = app.installed_models.iter().any(|(_, name, _)| name == meta.filename);
        let status_icon = if is_installed { "✔" } else { "☁" };
        let status_style = if is_installed {
            Style::default().fg(app.theme.neon_green).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(app.theme.text_muted)
        };

        catalog_items.push(ListItem::new(vec![
            Line::from(vec![
                Span::styled(format!("{} ", status_icon), status_style),
                Span::styled(meta.id, Style::default().fg(app.theme.text_bright).add_modifier(Modifier::BOLD)),
                Span::styled(format!(" ({:.1}G)", meta.size_gb), Style::default().fg(app.theme.text_muted)),
            ]),
            Line::from(vec![
                Span::styled(format!("   {}", meta.category), Style::default().fg(app.theme.neon_cyan)),
            ]),
        ]));
    }

    let catalog_list = List::new(catalog_items)
        .block(
            Block::default()
                .title(" Model Vault (16GB Guide) ")
                .title_style(app.theme.title_style())
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(app.theme.border_dim))
                .style(Style::default().bg(app.theme.bg_card)),
        );
    f.render_widget(catalog_list, sidebar_chunks[2]);
}

fn draw_content_pane(f: &mut Frame, app: &mut App, area: Rect) {
    let pane_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(6),    // Output conversation
            Constraint::Length(6), // Input textarea
        ])
        .split(area);

    // 1. Output conversation with live syntax highlighted markdown
    let mut lines = Vec::new();

    if app.chat_history.is_empty() && app.current_stream.is_empty() {
        // Welcome splash
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled(" ⚡ Welcome to ", Style::default().fg(app.theme.text_dim)),
            Span::styled("Nirvana Code (v2)", Style::default().fg(app.theme.neon_cyan).add_modifier(Modifier::BOLD)),
            Span::styled(" - Next-Gen Apple Silicon Assistant", Style::default().fg(app.theme.text_dim)),
        ]));
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("  • ", Style::default().fg(app.theme.neon_green)),
            Span::styled("Quantized KV-Cache: ", Style::default().fg(app.theme.text_bright).add_modifier(Modifier::BOLD)),
            Span::styled("Q8_0 active, cutting KV RAM consumption by 50%.", Style::default().fg(app.theme.text_dim)),
        ]));
        lines.push(Line::from(vec![
            Span::styled("  • ", Style::default().fg(app.theme.neon_green)),
            Span::styled("Prefix State Reuse: ", Style::default().fg(app.theme.text_bright).add_modifier(Modifier::BOLD)),
            Span::styled("Cached system prompt yields sub-30ms Time-To-First-Token (TTFT).", Style::default().fg(app.theme.text_dim)),
        ]));
        lines.push(Line::from(vec![
            Span::styled("  • ", Style::default().fg(app.theme.neon_green)),
            Span::styled("Unified RAM Pinning: ", Style::default().fg(app.theme.text_bright).add_modifier(Modifier::BOLD)),
            Span::styled("mlock locks model weights in physical LPDDR5 RAM (zero paging).", Style::default().fg(app.theme.text_dim)),
        ]));
        lines.push(Line::from(vec![
            Span::styled("  • ", Style::default().fg(app.theme.neon_green)),
            Span::styled("Speculative Verification: ", Style::default().fg(app.theme.text_bright).add_modifier(Modifier::BOLD)),
            Span::styled("Fast draft verification pipeline with KV rollback.", Style::default().fg(app.theme.text_dim)),
        ]));
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("  Type your prompt below or press ", Style::default().fg(app.theme.text_muted)),
            Span::styled("Ctrl+K", Style::default().fg(app.theme.neon_amber).add_modifier(Modifier::BOLD)),
            Span::styled(" to select templates or switch models.", Style::default().fg(app.theme.text_muted)),
        ]));
    } else {
        // Render history items
        for msg in &app.chat_history {
            if msg.role == "user" {
                lines.push(Line::from(vec![
                    Span::styled("USER > ", Style::default().fg(app.theme.neon_magenta).add_modifier(Modifier::BOLD)),
                    Span::styled(&msg.content, Style::default().fg(app.theme.text_bright)),
                ]));
                lines.push(Line::from(""));
            } else {
                let prefix_badge = if msg.prefix_hit {
                    format!(" [Prefix Hit: +{} tok]", msg.prefix_reused.unwrap_or(0))
                } else {
                    "".to_string()
                };
                let stat_header = format!(
                    "NIRVANA ASSISTANT [TTFT: {}ms | {:.1} tok/s]{}",
                    msg.ttft_ms.unwrap_or(0),
                    msg.tps.unwrap_or(0.0),
                    prefix_badge
                );
                lines.push(Line::from(vec![
                    Span::styled(stat_header, Style::default().fg(app.theme.neon_cyan).add_modifier(Modifier::BOLD)),
                ]));
                lines.extend(SyntaxHighlighter::render_markdown_lines(&msg.content));
                lines.push(Line::from(""));
            }
        }

        // Render live streaming response
        if !app.current_stream.is_empty() {
            let prefix_badge = if app.current_prefix_hit {
                format!(" [Prefix Hit: +{} tok]", app.current_prefix_reused)
            } else {
                "".to_string()
            };
            let live_header = format!("NIRVANA GENERATING...{}", prefix_badge);
            lines.push(Line::from(vec![
                Span::styled(live_header, Style::default().fg(app.theme.neon_green).add_modifier(Modifier::BOLD)),
            ]));
            lines.extend(SyntaxHighlighter::render_markdown_lines(&app.current_stream));
        }
    }

    let out_block = Block::default()
        .title(" Terminal Workspace ")
        .title_style(app.theme.title_style())
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(app.theme.border_dim))
        .style(Style::default().bg(app.theme.bg_dark));

    let out_paragraph = Paragraph::new(lines)
        .block(out_block)
        .wrap(Wrap { trim: false })
        .scroll((app.scroll_offset, 0));
    f.render_widget(out_paragraph, pane_chunks[0]);

    // 2. Input Box
    let is_generating = app.engine_state == EngineState::Generating;
    let input_title = if is_generating {
        " Prompt (Generating... Press Ctrl+C to cancel) "
    } else {
        " Prompt / Code Question (Ctrl+S to Send | Ctrl+K for Palette) "
    };

    let input_border_style = if is_generating {
        Style::default().fg(app.theme.neon_amber)
    } else {
        Style::default().fg(app.theme.border_focus)
    };

    let input_block = Block::default()
        .title(input_title)
        .title_style(Style::default().fg(app.theme.neon_cyan).add_modifier(Modifier::BOLD))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(input_border_style)
        .style(Style::default().bg(app.theme.bg_input));

    app.input_textarea.set_block(input_block);
    f.render_widget(&app.input_textarea, pane_chunks[1]);
}

fn draw_footer_status(f: &mut Frame, app: &App, area: Rect) {
    let toast = app.toast_message.as_ref().map(|(msg, _)| msg.as_str()).unwrap_or("");

    let footer_line = Line::from(vec![
        Span::styled(" [Ctrl+S] ", Style::default().fg(app.theme.neon_cyan).add_modifier(Modifier::BOLD)),
        Span::styled("Send  ", Style::default().fg(app.theme.text_dim)),
        Span::styled("[Ctrl+K] ", Style::default().fg(app.theme.neon_amber).add_modifier(Modifier::BOLD)),
        Span::styled("Palette  ", Style::default().fg(app.theme.text_dim)),
        Span::styled("[Ctrl+Y] ", Style::default().fg(app.theme.neon_green).add_modifier(Modifier::BOLD)),
        Span::styled("Copy Code  ", Style::default().fg(app.theme.text_dim)),
        Span::styled("[Ctrl+C] ", Style::default().fg(app.theme.neon_magenta).add_modifier(Modifier::BOLD)),
        Span::styled("Cancel  ", Style::default().fg(app.theme.text_dim)),
        Span::styled("[Esc] ", Style::default().fg(app.theme.text_muted).add_modifier(Modifier::BOLD)),
        Span::styled("Clear  ", Style::default().fg(app.theme.text_dim)),
        Span::styled(format!("   {}", toast), Style::default().fg(app.theme.neon_green).add_modifier(Modifier::BOLD)),
    ]);

    let footer_p = Paragraph::new(footer_line).style(Style::default().bg(app.theme.bg_card));
    f.render_widget(footer_p, area);
}

fn draw_command_palette(f: &mut Frame, app: &App, area: Rect) {
    // Center popup rect
    let popup_width = 64;
    let popup_height = 18;
    let popup_x = (area.width.saturating_sub(popup_width)) / 2;
    let popup_y = (area.height.saturating_sub(popup_height)) / 2;
    let popup_rect = Rect::new(popup_x, popup_y, popup_width, popup_height);

    f.render_widget(Clear, popup_rect);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(4)])
        .split(popup_rect);

    // Search query box
    let query_line = Line::from(vec![
        Span::styled(" 🔍 ", Style::default().fg(app.theme.neon_cyan)),
        Span::styled(&app.palette_query, Style::default().fg(app.theme.text_bright).add_modifier(Modifier::BOLD)),
        Span::styled("█", Style::default().fg(app.theme.neon_cyan)),
    ]);
    let query_p = Paragraph::new(query_line).block(
        Block::default()
            .title(" Command Palette [Ctrl+K] ")
            .title_style(app.theme.title_style())
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(app.theme.border_focus))
            .style(Style::default().bg(app.theme.bg_card)),
    );
    f.render_widget(query_p, chunks[0]);

    // Filtered items list
    let filtered = PaletteManager::filter_items(&app.palette_items, &app.palette_query);
    let mut list_items = Vec::new();

    for (idx, item) in filtered.iter().enumerate() {
        let is_selected = idx == app.palette_index;
        let prefix = if is_selected { "▶ " } else { "  " };

        let title_style = if is_selected {
            Style::default().fg(app.theme.neon_cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(app.theme.text_bright)
        };

        let cat_badge = format!("[{}] ", item.category);

        list_items.push(ListItem::new(vec![
            Line::from(vec![
                Span::styled(prefix, Style::default().fg(app.theme.neon_cyan)),
                Span::styled(cat_badge, Style::default().fg(app.theme.neon_magenta)),
                Span::styled(&item.title, title_style),
            ]),
            Line::from(vec![
                Span::raw("    "),
                Span::styled(&item.subtitle, Style::default().fg(app.theme.text_muted)),
            ]),
        ]));
    }

    let items_list = List::new(list_items).block(
        Block::default()
            .borders(Borders::BOTTOM | Borders::LEFT | Borders::RIGHT)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(app.theme.border_focus))
            .style(Style::default().bg(app.theme.bg_card)),
    );
    f.render_widget(items_list, chunks[1]);
}
