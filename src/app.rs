use crate::clipboard::ClipboardHelper;
use crate::engine::{ModelEngine, StreamEvent};
use crate::hardware::SiliconProfile;
use crate::model_manager::ModelManager;
use crate::palette::{PaletteAction, PaletteItem, PaletteManager};
use crate::syntax::SyntaxHighlighter;
use crate::templates::{PromptTemplate, TEMPLATES};
use crate::theme::Theme;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};
use tui_textarea::TextArea;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum AppMode {
    Chat,
    PromptCraft,
    SiliconHUD,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub ttft_ms: Option<u128>,
    pub tps: Option<f64>,
    pub tokens: Option<usize>,
    pub prefix_reused: Option<usize>,
    pub prefix_hit: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineState {
    Idle,
    Generating,
    Error,
}

pub struct App<'a> {
    pub theme: Theme,
    pub mode: AppMode,
    pub hardware: SiliconProfile,
    pub model_path: PathBuf,
    pub model_name: String,
    pub engine: Arc<ModelEngine>,
    pub engine_state: EngineState,
    pub active_template: &'static PromptTemplate,
    pub chat_history: Vec<ChatMessage>,
    pub current_stream: String,
    pub current_ttft_ms: Option<u128>,
    pub current_tps: Option<f64>,
    pub current_tokens: usize,
    pub current_prefix_reused: usize,
    pub current_prefix_hit: bool,
    pub input_textarea: TextArea<'a>,
    pub cancel_token: Option<Arc<AtomicBool>>,
    pub stream_rx: Option<UnboundedReceiver<StreamEvent>>,
    pub toast_message: Option<(String, std::time::Instant)>,
    pub show_palette: bool,
    pub palette_query: String,
    pub palette_index: usize,
    pub palette_items: Vec<PaletteItem>,
    pub installed_models: Vec<(PathBuf, String, u64)>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub scroll_offset: u16,
}

impl<'a> App<'a> {
    pub fn new(
        engine: Arc<ModelEngine>,
        model_path: PathBuf,
        max_tokens: usize,
        temperature: f32,
    ) -> Self {
        let hardware = SiliconProfile::detect();
        let installed_models = ModelManager::list_installed();
        let palette_items = PaletteManager::build_items(&installed_models);

        let model_name = model_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "Custom Model".to_string());

        let mut textarea = TextArea::default();
        textarea.set_placeholder_text("Type your prompt or code question... (Enter to newline, Ctrl+S to send, Ctrl+K for palette)");

        Self {
            theme: Theme::default(),
            mode: AppMode::Chat,
            hardware,
            model_path,
            model_name,
            engine,
            engine_state: EngineState::Idle,
            active_template: &TEMPLATES[0],
            chat_history: Vec::new(),
            current_stream: String::new(),
            current_ttft_ms: None,
            current_tps: None,
            current_tokens: 0,
            current_prefix_reused: 0,
            current_prefix_hit: false,
            input_textarea: textarea,
            cancel_token: None,
            stream_rx: None,
            toast_message: None,
            show_palette: false,
            palette_query: String::new(),
            palette_index: 0,
            palette_items,
            installed_models,
            max_tokens,
            temperature,
            scroll_offset: 0,
        }
    }

    pub fn set_toast(&mut self, msg: &str) {
        self.toast_message = Some((msg.to_string(), std::time::Instant::now()));
    }

    pub fn update_toast(&mut self) {
        if let Some((_, time)) = &self.toast_message {
            if time.elapsed().as_secs() > 3 {
                self.toast_message = None;
            }
        }
    }

    pub fn send_input(&mut self) {
        if self.engine_state == EngineState::Generating {
            return;
        }

        let input_text = self.input_textarea.lines().join("\n").trim().to_string();
        if input_text.is_empty() {
            return;
        }

        // Clear textarea
        self.input_textarea = TextArea::default();
        self.input_textarea.set_placeholder_text("Type your prompt or code question... (Ctrl+S to send, Ctrl+K for palette)");

        // Add user message to history
        self.chat_history.push(ChatMessage {
            role: "user".to_string(),
            content: input_text.clone(),
            ttft_ms: None,
            tps: None,
            tokens: None,
            prefix_reused: None,
            prefix_hit: false,
        });

        // Format prompt according to active template
        let full_prompt = match self.mode {
            AppMode::Chat => {
                // Multi-turn context representation
                let mut prompt_ctx = format!(
                    "<|im_start|>system\n{}<|im_end|>\n",
                    self.active_template.system_prompt
                );
                for msg in &self.chat_history {
                    prompt_ctx.push_str(&format!(
                        "<|im_start|>{}\n{}<|im_end|>\n",
                        msg.role, msg.content
                    ));
                }
                prompt_ctx.push_str("<|im_start|>assistant\n");
                prompt_ctx
            }
            _ => self.active_template.build_full_context(&input_text),
        };

        // Reset stream state
        self.current_stream.clear();
        self.current_ttft_ms = None;
        self.current_tps = None;
        self.current_tokens = 0;
        self.current_prefix_reused = 0;
        self.current_prefix_hit = false;
        self.engine_state = EngineState::Generating;
        self.scroll_offset = 0;

        let (tx, rx) = unbounded_channel();
        self.stream_rx = Some(rx);

        let cancel_token = Arc::new(AtomicBool::new(false));
        self.cancel_token = Some(cancel_token.clone());

        let engine = self.engine.clone();
        let max_tokens = self.max_tokens;
        let temperature = self.temperature;

        // Spawn inference generation on blocking background thread
        tokio::task::spawn_blocking(move || {
            let _ = engine.stream_generate(
                &full_prompt,
                max_tokens,
                temperature,
                cancel_token,
                tx,
            );
        });
    }

    pub fn cancel_generation(&mut self) {
        if let Some(token) = &self.cancel_token {
            token.store(true, Ordering::Relaxed);
        }
        self.engine_state = EngineState::Idle;
        self.set_toast("⏹ Generation stopped");
    }

    pub fn poll_stream(&mut self) {
        if let Some(rx) = &mut self.stream_rx {
            while let Ok(event) = rx.try_recv() {
                match event {
                    StreamEvent::Token(piece) => {
                        self.current_stream.push_str(&piece);
                    }
                    StreamEvent::Stats {
                        ttft_ms,
                        tokens_per_sec,
                        total_tokens,
                        prefix_tokens_reused,
                        prefix_cache_hit,
                        ..
                    } => {
                        self.current_ttft_ms = Some(ttft_ms);
                        self.current_tps = Some(tokens_per_sec);
                        self.current_tokens = total_tokens;
                        self.current_prefix_reused = prefix_tokens_reused;
                        self.current_prefix_hit = prefix_cache_hit;
                    }
                    StreamEvent::Done => {
                        self.engine_state = EngineState::Idle;
                        // Save assistant response to history
                        self.chat_history.push(ChatMessage {
                            role: "assistant".to_string(),
                            content: self.current_stream.clone(),
                            ttft_ms: self.current_ttft_ms,
                            tps: self.current_tps,
                            tokens: Some(self.current_tokens),
                            prefix_reused: Some(self.current_prefix_reused),
                            prefix_hit: self.current_prefix_hit,
                        });
                        break;
                    }
                    StreamEvent::Error(err) => {
                        self.engine_state = EngineState::Error;
                        self.set_toast(&format!("❌ {}", err));
                        break;
                    }
                }
            }
        }
    }

    pub fn copy_last_response(&mut self) {
        let text_to_copy = if !self.current_stream.is_empty() {
            &self.current_stream
        } else if let Some(msg) = self.chat_history.iter().rev().find(|m| m.role == "assistant") {
            &msg.content
        } else {
            self.set_toast("No response to copy");
            return;
        };

        if ClipboardHelper::copy_text(text_to_copy).is_ok() {
            self.set_toast("✔ Copied full response to clipboard!");
        } else {
            self.set_toast("❌ Failed to copy to clipboard");
        }
    }

    pub fn copy_first_code_snippet(&mut self) {
        let content = if !self.current_stream.is_empty() {
            &self.current_stream
        } else if let Some(msg) = self.chat_history.iter().rev().find(|m| m.role == "assistant") {
            &msg.content
        } else {
            self.set_toast("No code snippet available");
            return;
        };

        let snippets = SyntaxHighlighter::extract_code_blocks(content);
        if let Some(first) = snippets.first() {
            if ClipboardHelper::copy_text(first).is_ok() {
                self.set_toast("✔ Copied code block to clipboard!");
            }
        } else {
            self.set_toast("No code blocks detected in response");
        }
    }

    pub fn execute_palette_action(&mut self, action: PaletteAction) {
        match action {
            PaletteAction::SelectTemplate(id) => {
                if let Some(tmpl) = TEMPLATES.iter().find(|t| t.id == id) {
                    self.active_template = tmpl;
                    self.set_toast(&format!("✔ Switched template: {}", tmpl.name));
                }
            }
            PaletteAction::SelectModel(path_str) => {
                self.set_toast(&format!("Restart with --model to load {}", path_str));
            }
            PaletteAction::ClearHistory => {
                self.chat_history.clear();
                self.current_stream.clear();
                self.engine.clear_cache();
                self.set_toast("✔ KV Cache & conversation history cleared!");
            }
            PaletteAction::CopyLastResponse => {
                self.copy_last_response();
            }
            PaletteAction::CopyCodeSnippet(_) => {
                self.copy_first_code_snippet();
            }
            PaletteAction::ToggleSpeculative => {
                self.set_toast("Speculative mode can be launched via: nirvana-code --speculative");
            }
            PaletteAction::ToggleKvQuantization => {
                self.set_toast("KV-Cache quantization active: Q8_0 [50% Unified RAM Saved]");
            }
            PaletteAction::OpenDocs => {
                self.set_toast("Visit atomic.chat/blog/guides/best-local-llm-16gb");
            }
            PaletteAction::Quit => {}
        }
        self.show_palette = false;
    }
}
