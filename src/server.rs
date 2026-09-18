use anyhow::Result;
use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        Html, IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::unbounded_channel;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;

use crate::attachment::Attachment;
use crate::engine::{GenerationConfig, KvQuantMode, ModelEngine, StreamEvent};
use crate::model_manager::ModelManager;

pub struct ServerEngineInner {
    pub engine: Arc<ModelEngine>,
    pub model_name: String,
    pub model_path: PathBuf,
}

#[derive(Clone)]
pub struct ServerState {
    pub inner: Arc<RwLock<ServerEngineInner>>,
    pub gpu_layers: u32,
    pub use_mlock: bool,
    pub kv_mode: KvQuantMode,
    pub ctx_size: u32,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AttachmentPayloadDto {
    pub filename: String,
    #[serde(default)]
    pub data: Option<String>,
    #[serde(default)]
    pub extracted_text: Option<String>,
    #[serde(default)]
    pub metadata_summary: Option<String>,
    #[serde(default)]
    pub file_type: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ProcessAttachmentRequest {
    pub filename: String,
    pub data: String,
}

#[derive(Debug, Serialize)]
pub struct ProcessAttachmentResponse {
    pub status: String,
    pub filename: String,
    pub file_type: String,
    pub size_bytes: u64,
    pub metadata_summary: String,
    pub extracted_text_preview: String,
    pub extracted_text: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[allow(dead_code)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessageDto>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_min_p")]
    pub min_p: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_top_k")]
    pub top_k: i32,
    #[serde(default)]
    pub ngram_speculative: bool,
    #[serde(default)]
    pub attachment: Option<AttachmentPayloadDto>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessageDto {
    pub role: String,
    pub content: String,
}

fn default_temperature() -> f32 {
    0.7
}
fn default_max_tokens() -> usize {
    2048
}
fn default_min_p() -> f32 {
    0.05
}
fn default_top_p() -> f32 {
    0.9
}
fn default_top_k() -> i32 {
    40
}

#[derive(Debug, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
    pub active: bool,
    pub size_bytes: u64,
    pub display_name: String,
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: &'static str,
    pub current_model: String,
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    pub model: String,
}

#[derive(Debug, Serialize)]
pub struct LoadModelResponse {
    pub status: String,
    pub model: String,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: UsageInfo,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessageDto,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct UsageInfo {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

pub fn create_router(state: ServerState) -> Router {
    Router::new()
        .route("/", get(handle_index))
        .route("/index.html", get(handle_index))
        .route("/v1/models", get(handle_models))
        .route("/v1/models/load", post(handle_load_model))
        .route("/v1/attachments/process", post(handle_process_attachment))
        .route("/v1/chat/completions", post(handle_chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn handle_process_attachment(
    Json(req): Json<ProcessAttachmentRequest>,
) -> Response {
    let filename = req.filename.trim().to_string();
    let data = req.data.trim().to_string();
    if filename.is_empty() || data.is_empty() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "Filename and data payload cannot be empty."
            })),
        )
            .into_response();
    }

    let fname_clone = filename.clone();
    let data_clone = data.clone();
    let process_res = tokio::task::spawn_blocking(move || {
        if data_clone.starts_with("data:") || (data_clone.len() > 100 && !data_clone.contains('\n') && !data_clone.starts_with('/')) {
            Attachment::from_base64(&fname_clone, &data_clone)
        } else {
            Attachment::from_file(&data_clone).or_else(|_| Attachment::from_base64(&fname_clone, &data_clone))
        }
    })
    .await;

    match process_res {
        Ok(Ok(att)) => {
            let preview = if att.extracted_text.len() > 400 {
                format!("{}...", &att.extracted_text[..400])
            } else {
                att.extracted_text.clone()
            };

            Json(ProcessAttachmentResponse {
                status: "success".to_string(),
                filename: att.filename,
                file_type: att.file_type.label().to_string(),
                size_bytes: att.size_bytes,
                metadata_summary: att.metadata_summary,
                extracted_text_preview: preview,
                extracted_text: att.extracted_text,
            })
            .into_response()
        }
        Ok(Err(e)) => (
            axum::http::StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "error": format!("Attachment extraction failed: {}", e)
            })),
        )
            .into_response(),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Attachment task panicked: {}", e)
            })),
        )
            .into_response(),
    }
}

async fn handle_index() -> Html<&'static str> {
    Html(include_str!("web/index.html"))
}

async fn handle_models(State(state): State<ServerState>) -> Json<ModelListResponse> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let inner = state.inner.read().await;
    let current_name = inner.model_name.clone();
    let current_path = inner.model_path.clone();
    drop(inner);

    let installed = ModelManager::list_installed();
    let mut data = Vec::new();
    let mut found_current = false;

    for (path, name, size) in &installed {
        let is_active = *path == current_path || *name == current_name;
        if is_active {
            found_current = true;
        }
        data.push(ModelCard {
            id: name.clone(),
            object: "model",
            created: now,
            owned_by: "nirvana",
            active: is_active,
            size_bytes: *size,
            display_name: name.clone(),
        });
    }

    if !found_current {
        data.insert(
            0,
            ModelCard {
                id: current_name.clone(),
                object: "model",
                created: now,
                owned_by: "nirvana",
                active: true,
                size_bytes: 0,
                display_name: current_name.clone(),
            },
        );
    }

    Json(ModelListResponse {
        object: "list",
        current_model: current_name,
        data,
    })
}

async fn handle_load_model(
    State(state): State<ServerState>,
    Json(req): Json<LoadModelRequest>,
) -> Response {
    let target = req.model.trim();
    if target.is_empty() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "Model identifier cannot be empty."
            })),
        )
            .into_response();
    }

    let resolved_path = match ModelManager::resolve_model_path(Some(Path::new(target))) {
        Some(p) => p,
        None => {
            return (
                axum::http::StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": format!("Model '{}' not found among installed models. Run 'nirvana-code download {}' first.", target, target)
                })),
            )
                .into_response();
        }
    };

    let model_name = resolved_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| target.to_string());

    {
        let inner = state.inner.read().await;
        if inner.model_path == resolved_path {
            return Json(LoadModelResponse {
                status: "ok".to_string(),
                model: model_name,
                path: resolved_path.display().to_string(),
                message: "Model is already active on Apple Silicon Metal GPU".to_string(),
            })
            .into_response();
        }
    }

    println!("⚡ Dynamic Model Switch: Loading {} into Apple Silicon Metal GPU...", model_name);

    let gpu_layers = state.gpu_layers;
    let use_mlock = state.use_mlock;
    let kv_mode = state.kv_mode;
    let ctx_size = state.ctx_size;
    let path_clone = resolved_path.clone();

    let load_res = tokio::task::spawn_blocking(move || {
        ModelEngine::load(&path_clone, gpu_layers, use_mlock, kv_mode, ctx_size)
    })
    .await;

    match load_res {
        Ok(Ok(new_engine)) => {
            let mut inner = state.inner.write().await;
            inner.engine = Arc::new(new_engine);
            inner.model_name = model_name.clone();
            inner.model_path = resolved_path.clone();
            drop(inner);

            println!("✔ Switched active model to {} (Metal GPU Ready)", model_name);

            Json(LoadModelResponse {
                status: "ok".to_string(),
                model: model_name,
                path: resolved_path.display().to_string(),
                message: "Model successfully loaded onto Metal GPU".to_string(),
            })
            .into_response()
        }
        Ok(Err(e)) => {
            eprintln!("❌ Failed to load model: {}", e);
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Failed to load model: {}", e)
                })),
            )
                .into_response()
        }
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Model load task panicked: {}", e)
            })),
        )
            .into_response(),
    }
}

fn format_messages_to_prompt(
    messages: &[ChatMessageDto],
    attachment: Option<&AttachmentPayloadDto>,
) -> String {
    let mut prompt = String::new();
    let mut has_system = false;

    for msg in messages {
        if msg.role.to_lowercase() == "system" {
            prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", msg.content));
            has_system = true;
        }
    }

    if !has_system {
        prompt.push_str("<|im_start|>system\nYou are Nirvana Code, an ultra-low latency Apple Silicon coding assistant. Provide clean, fast, reliable code.<|im_end|>\n");
    }

    let non_sys: Vec<&ChatMessageDto> = messages
        .iter()
        .filter(|m| m.role.to_lowercase() != "system")
        .collect();
    let count = non_sys.len();

    for (i, msg) in non_sys.into_iter().enumerate() {
        if i + 1 == count && msg.role.to_lowercase() == "user" && attachment.is_some() {
            let att = attachment.unwrap();
            let mut formatted_content = String::new();
            formatted_content.push_str(&format!(
                "[ATTACHED FILE: {} | Type: {} | {}]\n",
                att.filename,
                att.file_type.as_deref().unwrap_or("FILE"),
                att.metadata_summary.as_deref().unwrap_or("")
            ));
            if let Some(ref text) = att.extracted_text {
                formatted_content.push_str("--- BEGIN ATTACHED CONTENT ---\n");
                formatted_content.push_str(text);
                if !text.ends_with('\n') {
                    formatted_content.push('\n');
                }
                formatted_content.push_str("--- END ATTACHED CONTENT ---\n\n");
            }
            if msg.content.trim().is_empty() {
                formatted_content.push_str(&format!(
                    "Please analyze the attached {} (`{}`) and provide a detailed explanation of its contents and key insights.",
                    att.file_type.as_deref().unwrap_or("file").to_lowercase(),
                    att.filename
                ));
            } else {
                formatted_content.push_str(&msg.content);
            }

            prompt.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                msg.role, formatted_content
            ));
        } else {
            prompt.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                msg.role, msg.content
            ));
        }
    }

    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

async fn handle_chat_completions(
    State(state): State<ServerState>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Response {
    // Check if request asks for a specific installed model that isn't currently loaded
    if let Some(ref req_model) = payload.model {
        let req_trim = req_model.trim();
        if !req_trim.is_empty() && req_trim != "nirvana-code" {
            let current_differs = {
                let inner = state.inner.read().await;
                inner.model_name != req_trim
                    && !inner.model_name.contains(req_trim)
                    && !req_trim.contains(&inner.model_name)
            };

            if current_differs {
                if let Some(new_path) = ModelManager::resolve_model_path(Some(Path::new(req_trim))) {
                    let should_load = {
                        let inner = state.inner.read().await;
                        inner.model_path != new_path
                    };

                    if should_load {
                        let gpu_layers = state.gpu_layers;
                        let use_mlock = state.use_mlock;
                        let kv_mode = state.kv_mode;
                        let ctx_size = state.ctx_size;
                        let path_clone = new_path.clone();

                        let load_res = tokio::task::spawn_blocking(move || {
                            ModelEngine::load(&path_clone, gpu_layers, use_mlock, kv_mode, ctx_size)
                        })
                        .await;

                        if let Ok(Ok(new_engine)) = load_res {
                            let mut inner = state.inner.write().await;
                            inner.model_name = new_path
                                .file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_else(|| req_trim.to_string());
                            inner.model_path = new_path;
                            inner.engine = Arc::new(new_engine);
                        }
                    }
                }
            }
        }
    }

    let (engine, active_model_name) = {
        let inner = state.inner.read().await;
        (inner.engine.clone(), inner.model_name.clone())
    };

    let mut attachment = payload.attachment.clone();
    if let Some(ref mut att) = attachment {
        if att.extracted_text.is_none() {
            if let Some(ref data) = att.data {
                let filename = att.filename.clone();
                let data = data.clone();
                let parse_res = tokio::task::spawn_blocking(move || {
                    if data.starts_with("data:") || (data.len() > 100 && !data.contains('\n') && !data.starts_with('/')) {
                        Attachment::from_base64(&filename, &data)
                    } else {
                        Attachment::from_file(&data).or_else(|_| Attachment::from_base64(&filename, &data))
                    }
                })
                .await;

                if let Ok(Ok(parsed)) = parse_res {
                    att.extracted_text = Some(parsed.extracted_text);
                    att.metadata_summary = Some(parsed.metadata_summary);
                    att.file_type = Some(parsed.file_type.label().to_string());
                }
            }
        }
    }

    let prompt = format_messages_to_prompt(&payload.messages, attachment.as_ref());
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let req_id = format!("chatcmpl-{}", now);

    let config = GenerationConfig {
        max_tokens: payload.max_tokens,
        temperature: payload.temperature,
        min_p: payload.min_p,
        top_p: payload.top_p,
        top_k: payload.top_k,
        use_ngram_speculative: payload.ngram_speculative,
    };

    let (tx, mut rx) = unbounded_channel();
    let cancel = Arc::new(AtomicBool::new(false));

    let prompt_clone = prompt.clone();
    let cancel_clone = cancel.clone();
    let engine_clone = engine.clone();

    tokio::task::spawn_blocking(move || {
        let _ = engine_clone.stream_generate_with_config(&prompt_clone, &config, cancel_clone, tx);
    });

    if payload.stream {
        // Stream SSE tokens
        let model_name_for_stream = active_model_name.clone();
        let stream = async_stream::stream! {
            // First chunk with role
            let initial_chunk = ChatCompletionChunk {
                id: req_id.clone(),
                object: "chat.completion.chunk",
                created: now,
                model: model_name_for_stream.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: Some("assistant".to_string()),
                        content: None,
                    },
                    finish_reason: None,
                }],
            };
            let json = serde_json::to_string(&initial_chunk).unwrap_or_default();
            yield Ok::<Event, Infallible>(Event::default().data(json));

            while let Some(event) = rx.recv().await {
                match event {
                    StreamEvent::Token(token_str) => {
                        let chunk = ChatCompletionChunk {
                            id: req_id.clone(),
                            object: "chat.completion.chunk",
                            created: now,
                            model: model_name_for_stream.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: Some(token_str),
                                },
                                finish_reason: None,
                            }],
                        };
                        let json = serde_json::to_string(&chunk).unwrap_or_default();
                        yield Ok(Event::default().data(json));
                    }
                    StreamEvent::Done { .. } => {
                        let final_chunk = ChatCompletionChunk {
                            id: req_id.clone(),
                            object: "chat.completion.chunk",
                            created: now,
                            model: model_name_for_stream.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: None,
                                },
                                finish_reason: Some("stop"),
                            }],
                        };
                        let json = serde_json::to_string(&final_chunk).unwrap_or_default();
                        yield Ok(Event::default().data(json));
                        yield Ok(Event::default().data("[DONE]"));
                        break;
                    }
                    _ => {}
                }
            }
        };

        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let mut full_content = String::new();
        let prompt_toks = 0;
        let mut completion_toks = 0;

        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(token_str) => {
                    full_content.push_str(&token_str);
                }
                StreamEvent::Stats { total_tokens, .. } => {
                    completion_toks = total_tokens;
                }
                StreamEvent::Done => {
                    break;
                }
                _ => {}
            }
        }

        let resp = ChatCompletionResponse {
            id: req_id,
            object: "chat.completion",
            created: now,
            model: active_model_name,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessageDto {
                    role: "assistant".to_string(),
                    content: full_content,
                },
                finish_reason: "stop",
            }],
            usage: UsageInfo {
                prompt_tokens: prompt_toks,
                completion_tokens: completion_toks,
                total_tokens: prompt_toks + completion_toks,
            },
        };

        Json(resp).into_response()
    }
}

pub async fn run_server(
    engine: Arc<ModelEngine>,
    model_name: String,
    model_path: PathBuf,
    host: &str,
    port: u16,
    socket_path: Option<PathBuf>,
    gpu_layers: u32,
    use_mlock: bool,
    kv_mode: KvQuantMode,
    ctx_size: u32,
) -> Result<()> {
    let inner = Arc::new(RwLock::new(ServerEngineInner {
        engine,
        model_name: model_name.clone(),
        model_path,
    }));

    let state = ServerState {
        inner,
        gpu_layers,
        use_mlock,
        kv_mode,
        ctx_size,
    };

    let app = create_router(state);

    println!("\n⚡ [NIRVANA CODE] Web UI & OpenAI-Compatible Local API Server");
    println!("   Platform:        Apple Silicon Metal 3 Unified LPDDR5");
    println!("   Model:           {}", model_name);
    println!("   Web Interface:   http://{}:{}", host, port);
    println!("   Chat API URL:    http://{}:{}/v1/chat/completions", host, port);
    println!("   Models API URL:  http://{}:{}/v1/models", host, port);
    println!("   Switch API URL:  POST http://{}:{}/v1/models/load", host, port);

    // If Unix socket is configured
    if let Some(ref sock) = socket_path {
        if sock.exists() {
            let _ = std::fs::remove_file(sock);
        }
        println!("   Unix Socket:     {}", sock.display());
        let unix_listener = tokio::net::UnixListener::bind(sock)?;
        let app_clone = app.clone();
        tokio::spawn(async move {
            let _ = axum::serve(unix_listener, app_clone).await;
        });
    }

    println!("\n   Ready for Web Browser, VS Code Continue, Cursor, Neovim, and OpenAI SDKs!");
    println!("   Press Ctrl+C to stop.\n");

    let addr = format!("{}:{}", host, port);
    let tcp_listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(tcp_listener, app).await?;

    Ok(())
}
