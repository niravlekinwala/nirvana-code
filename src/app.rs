use crate::clipboard::ClipboardHelper;
use crate::engine::{GenerationConfig, ModelEngine, StreamEvent};
use crate::hardware::SiliconProfile;
use crate::model_manager::ModelManager;
use crate::palette::{PaletteAction, PaletteItem, PaletteManager};
use crate::syntax::SyntaxHighlighter;
use crate::templates::{PromptTemplate, TEMPLATES};
use crate::theme::Theme;
use anyhow::Result;
use std::path::{Path, PathBuf};
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
    pub min_p: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub ngram_speculative: bool,
    pub scroll_offset: u16,
    pub max_scroll: u16,
    pub auto_scroll: bool,
    pub exit_confirmation: bool,
    pub exit_confirmation_time: Option<std::time::Instant>,
    pub show_sidebar: bool,
}

impl<'a> App<'a> {
    pub fn new(
        engine: Arc<ModelEngine>,
        model_path: PathBuf,
        max_tokens: usize,
        temperature: f32,
        min_p: f32,
        top_p: f32,
        top_k: i32,
        ngram_speculative: bool,
    ) -> Self {
        let hardware = SiliconProfile::detect();
        let installed_models = ModelManager::list_installed();
        let palette_items = PaletteManager::build_items(&installed_models);

        let model_name = model_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "Custom Model".to_string());

        let mut textarea = TextArea::default();
        textarea.set_placeholder_text("Type your prompt or code question... (Enter to send, Shift+Enter for newline, Ctrl+K for palette)");

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
            min_p,
            top_p,
            top_k,
            ngram_speculative,
            scroll_offset: 0,
            max_scroll: 0,
            auto_scroll: true,
            exit_confirmation: false,
            exit_confirmation_time: None,
            show_sidebar: true,
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
        if let Some(time) = self.exit_confirmation_time {
            if time.elapsed().as_secs() > 4 {
                self.exit_confirmation = false;
                self.exit_confirmation_time = None;
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

        // Clear textarea & reset auto scroll to follow new generation
        self.input_textarea = TextArea::default();
        self.input_textarea.set_placeholder_text("Type your prompt or code question... (Enter to send, Shift+Enter for newline, Ctrl+K for palette)");
        self.auto_scroll = true;
        self.exit_confirmation = false;

        // Check for slash commands (/model, /models, /clear, /sidebar)
        if input_text.starts_with('/') {
            let parts: Vec<&str> = input_text.split_whitespace().collect();
            let cmd = parts[0].to_lowercase();
            match cmd.as_str() {
                "/model" | "/switch" | "/load" => {
                    if parts.len() > 1 {
                        let target = parts[1..].join(" ");
                        if let Some(path) = ModelManager::resolve_model_path(Some(Path::new(&target))) {
                            let _ = self.switch_model(path);
                        } else {
                            self.set_toast(&format!("❌ Model '{}' not found. Press Ctrl+P to see installed models.", target));
                        }
                    } else {
                        self.show_palette = true;
                        self.palette_query = "Model:".to_string();
                        self.palette_index = 0;
                    }
                    return;
                }
                "/models" => {
                    self.show_palette = true;
                    self.palette_query = "Model:".to_string();
                    self.palette_index = 0;
                    return;
                }
                "/sidebar" => {
                    self.show_sidebar = !self.show_sidebar;
                    self.set_toast(if self.show_sidebar { "✔ Sidebar visible" } else { "✔ Sidebar hidden (Full Workspace)" });
                    return;
                }
                "/clear" => {
                    self.chat_history.clear();
                    self.current_stream.clear();
                    self.engine.clear_cache();
                    self.set_toast("✔ KV Cache & conversation history cleared");
                    return;
                }
                _ => {}
            }
        }

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
        let config = GenerationConfig {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            min_p: self.min_p,
            top_p: self.top_p,
            top_k: self.top_k,
            use_ngram_speculative: self.ngram_speculative,
        };

        // Spawn inference generation on blocking background thread
        tokio::task::spawn_blocking(move || {
            let _ = engine.stream_generate_with_config(
                &full_prompt,
                &config,
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
                        self.current_stream.clear();
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

    pub fn copy_full_conversation(&mut self) {
        if self.chat_history.is_empty() && self.current_stream.is_empty() {
            self.set_toast("No conversation to copy");
            return;
        }

        let mut full_text = String::new();
        for msg in &self.chat_history {
            if msg.role == "user" {
                full_text.push_str(&format!("### USER\n{}\n\n", msg.content));
            } else {
                full_text.push_str(&format!("### ASSISTANT\n{}\n\n", msg.content));
            }
        }
        if !self.current_stream.is_empty() {
            full_text.push_str(&format!("### ASSISTANT (streaming)\n{}\n\n", self.current_stream));
        }

        if ClipboardHelper::copy_text(full_text.trim()).is_ok() {
            self.set_toast("✔ Copied full conversation to clipboard!");
        } else {
            self.set_toast("❌ Failed to copy to clipboard");
        }
    }

    pub fn switch_model(&mut self, target_path: PathBuf) -> Result<()> {
        if self.model_path == target_path {
            self.set_toast(&format!("✔ Already using {}", self.model_name));
            return Ok(());
        }

        let filename = target_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".to_string());

        self.set_toast(&format!("⏳ Loading {} onto Metal GPU...", filename));

        // 1. Cancel active generation & clear KV cache
        self.cancel_generation();
        self.engine.clear_cache();

        let gpu_layers = self.engine.n_gpu_layers;
        let use_mlock = self.engine.use_mlock;
        let kv_mode = self.engine.kv_mode;
        let ctx_size = self.engine.n_ctx;

        match ModelEngine::load(&target_path, gpu_layers, use_mlock, kv_mode, ctx_size) {
            Ok(new_engine) => {
                self.engine = Arc::new(new_engine);
                self.model_path = target_path.clone();
                self.model_name = filename.clone();
                self.chat_history.clear();
                self.current_stream.clear();
                self.set_toast(&format!("✔ Active Model: {} (Metal GPU Ready)", filename));
                Ok(())
            }
            Err(e) => {
                self.set_toast(&format!("❌ Failed to load model: {}", e));
                Err(e)
            }
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
                let path = PathBuf::from(&path_str);
                let _ = self.switch_model(path);
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
            PaletteAction::CopyFullConversation => {
                self.copy_full_conversation();
            }
            PaletteAction::ToggleSidebar => {
                self.show_sidebar = !self.show_sidebar;
                self.set_toast(if self.show_sidebar {
                    "✔ Sidebar visible"
                } else {
                    "✔ Sidebar hidden (Full Workspace)"
                });
            }
            PaletteAction::CopyCodeSnippet(_) => {
                self.copy_first_code_snippet();
            }
            PaletteAction::ToggleSpeculative => {
                self.set_toast("Speculative mode can be launched via: nirvana-code --speculative");
            }
            PaletteAction::ToggleKvQuantization => {
                self.set_toast("KV-Cache active: F16 [Peak Speed 120+ tok/s]");
            }
            PaletteAction::OpenDocs => {
                self.set_toast("Visit atomic.chat/blog/guides/best-local-llm-16gb");
            }
            PaletteAction::Quit => {
                self.exit_confirmation = true;
                self.exit_confirmation_time = Some(std::time::Instant::now());
                self.set_toast("⚠️ Press Ctrl+C again to confirm exit");
            }
        }
        self.show_palette = false;
    }
}
