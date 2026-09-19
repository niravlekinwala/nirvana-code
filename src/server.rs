use anyhow::Result;
use axum::{
    extract::{DefaultBodyLimit, State},
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
use crate::engine::{GenerationConfig, InferenceEngine, KvQuantMode, StreamEvent};
use crate::model_manager::ModelManager;

pub struct ServerEngineInner {
    pub engine: InferenceEngine,
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
    pub active_cancel: Arc<tokio::sync::Mutex<Option<Arc<AtomicBool>>>>,
}

struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
    }
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
    pub seed: Option<u32>,
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
    pub current_ctx_size: u32,
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    pub model: String,
    #[serde(default)]
    pub ctx_size: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct ContextInfoResponse {
    pub status: String,
    pub ctx_size: u32,
    pub model: String,
    pub backend: String,
    pub kv_mode: String,
    pub p_cores: u32,
    pub gpu_layers: u32,
    pub mlock: bool,
}

#[derive(Debug, Deserialize)]
pub struct SetContextRequest {
    pub ctx_size: u32,
}

#[derive(Debug, Deserialize)]
pub struct ProjectScanRequest {
    #[serde(default)]
    pub path: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ProjectFileItem {
    pub relative_path: String,
    pub file_name: String,
    pub size_bytes: u64,
    pub extension: String,
    pub is_code: bool,
}

#[derive(Debug, Serialize)]
pub struct ProjectScanResponse {
    pub status: String,
    pub root_path: String,
    pub project_name: String,
    pub project_type: String,
    pub total_files: usize,
    pub files: Vec<ProjectFileItem>,
}

#[derive(Debug, Deserialize)]
pub struct ProjectFileRequest {
    pub root_path: String,
    pub file_path: String,
}

#[derive(Debug, Serialize)]
pub struct ProjectFileResponse {
    pub status: String,
    pub file_path: String,
    pub content: String,
    pub size_bytes: u64,
    pub line_count: usize,
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
        .route("/v1/context", get(handle_get_context).post(handle_set_context))
        .route("/v1/project/scan", post(handle_project_scan))
        .route("/v1/project/file", post(handle_project_file))
        .route("/v1/attachments/process", post(handle_process_attachment))
        .route("/v1/chat/completions", post(handle_chat_completions))
        .route("/v1/chat/stop", post(handle_chat_stop))
        .route("/v1/engine/reset", post(handle_engine_reset))
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024))
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
        if data_clone.starts_with("data:") || (!data_clone.starts_with('/') && !data_clone.starts_with('~') && !data_clone.starts_with('.')) {
            Attachment::from_base64(&fname_clone, &data_clone)
                .or_else(|_| Attachment::from_file(&data_clone))
        } else {
            Attachment::from_file(&data_clone)
                .or_else(|_| Attachment::from_base64(&fname_clone, &data_clone))
        }
    })
    .await;

    match process_res {
        Ok(Ok(att)) => {
            let preview = if att.extracted_text.chars().count() > 400 {
                let s: String = att.extracted_text.chars().take(400).collect();
                format!("{s}...")
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
    let current_ctx_size = inner.engine.n_ctx();
    drop(inner);

    let installed = ModelManager::list_installed();
    let mut data = Vec::new();
    let mut found_current = false;

    for (path, name, size) in &installed {
        let is_active = *path == current_path || *name == current_name;
        if is_active {
            found_current = true;
        }
        let format_tag = if ModelManager::is_mlx_model(path) {
            "MLX"
        } else {
            "GGUF"
        };
        data.push(ModelCard {
            id: name.clone(),
            object: "model",
            created: now,
            owned_by: format_tag,
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
        current_ctx_size,
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

    let ctx_target = req.ctx_size.unwrap_or(state.ctx_size);

    {
        let inner = state.inner.read().await;
        if inner.model_path == resolved_path {
            if let Some(new_ctx) = req.ctx_size {
                let _ = inner.engine.reconfigure_context(new_ctx);
            }
            return Json(LoadModelResponse {
                status: "ok".to_string(),
                model: model_name,
                path: resolved_path.display().to_string(),
                message: "Model is already active on Apple Silicon Metal GPU".to_string(),
            })
            .into_response();
        }
    }

    let is_mlx = ModelManager::is_mlx_model(&resolved_path);
    let backend_name = if is_mlx { "Apple MLX" } else { "Metal GPU" };
    println!("⚡ Dynamic Model Switch: Loading {model_name} into {backend_name}...");

    let gpu_layers = state.gpu_layers;
    let use_mlock = state.use_mlock;
    let kv_mode = state.kv_mode;
    let path_clone = resolved_path.clone();

    let load_res = tokio::task::spawn_blocking(move || {
        InferenceEngine::load(&path_clone, gpu_layers, use_mlock, kv_mode, ctx_target)
    })
    .await;

    match load_res {
        Ok(Ok(new_engine)) => {
            let badge = new_engine.backend_name();
            let mut inner = state.inner.write().await;
            inner.engine = new_engine;
            inner.model_name = model_name.clone();
            inner.model_path = resolved_path.clone();
            drop(inner);

            println!("✔ Switched active model to {model_name} ({badge} Ready)");

            Json(LoadModelResponse {
                status: "ok".to_string(),
                model: model_name,
                path: resolved_path.display().to_string(),
                message: format!("Model successfully loaded onto {badge}"),
            })
            .into_response()
        }
        Ok(Err(e)) => {
            eprintln!("❌ Failed to load model: {e}");
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

async fn handle_get_context(State(state): State<ServerState>) -> Json<ContextInfoResponse> {
    let inner = state.inner.read().await;
    let ctx_size = inner.engine.n_ctx();
    let model = inner.model_name.clone();
    let backend = inner.engine.backend_name().to_string();
    let kv_mode = inner.engine.kv_label();
    let p_cores = crate::hardware::SiliconProfile::detect().p_cores;
    let gpu_layers = state.gpu_layers;
    let mlock = state.use_mlock;

    Json(ContextInfoResponse {
        status: "ok".to_string(),
        ctx_size,
        model,
        backend,
        kv_mode,
        p_cores,
        gpu_layers,
        mlock,
    })
}

async fn handle_set_context(
    State(state): State<ServerState>,
    Json(req): Json<SetContextRequest>,
) -> Response {
    if req.ctx_size < 512 || req.ctx_size > 262144 {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "Context window must be between 512 and 262,144 tokens."
            })),
        )
            .into_response();
    }

    let inner = state.inner.read().await;
    let res = inner.engine.reconfigure_context(req.ctx_size);
    let model_name = inner.model_name.clone();
    drop(inner);

    match res {
        Ok(_) => {
            println!("⚡ User-defined Context Window: {} tokens (Model: {})", req.ctx_size, model_name);
            Json(serde_json::json!({
                "status": "ok",
                "ctx_size": req.ctx_size,
                "message": format!("Context window successfully set to {} tokens.", req.ctx_size)
            }))
            .into_response()
        }
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Failed to reconfigure context window: {}", e)
            })),
        )
            .into_response(),
    }
}

async fn handle_chat_stop(State(state): State<ServerState>) -> Response {
    let mut active = state.active_cancel.lock().await;
    let stopped = if let Some(token) = active.take() {
        token.store(true, std::sync::atomic::Ordering::Relaxed);
        true
    } else {
        false
    };
    Json(serde_json::json!({
        "status": "ok",
        "stopped": stopped,
        "message": if stopped { "Active generation stopped" } else { "No generation was active" }
    }))
    .into_response()
}

async fn handle_engine_reset(State(state): State<ServerState>) -> Response {
    let mut active = state.active_cancel.lock().await;
    if let Some(token) = active.take() {
        token.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    drop(active);

    let inner = state.inner.read().await;
    inner.engine.clear_cache();
    let model_name = inner.model_name.clone();
    drop(inner);

    println!("🛑 [Emergency Reset] Generation aborted and engine cache cleared ({model_name})");

    Json(serde_json::json!({
        "status": "ok",
        "message": "Engine reset and cache cleared successfully"
    }))
    .into_response()
}

async fn handle_project_scan(
    Json(req): Json<ProjectScanRequest>,
) -> Response {
    let root_path_buf = if let Some(ref p) = req.path {
        let trimmed = p.trim();
        if trimmed.is_empty() {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        } else {
            let path = PathBuf::from(trimmed);
            if path.is_relative() {
                std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")).join(path)
            } else {
                path
            }
        }
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    };

    let canonical_root = match root_path_buf.canonicalize() {
        Ok(c) => c,
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("Invalid project directory path: {}", e)
                })),
            )
                .into_response();
        }
    };

    if !canonical_root.is_dir() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "Target path is not a directory."
            })),
        )
            .into_response();
    }

    let project_name = canonical_root
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "project".to_string());

    let project_type = if canonical_root.join("Cargo.toml").exists() {
        "Rust (Cargo)"
    } else if canonical_root.join("package.json").exists() {
        "JavaScript / TypeScript (Node)"
    } else if canonical_root.join("pyproject.toml").exists() || canonical_root.join("requirements.txt").exists() {
        "Python"
    } else if canonical_root.join("go.mod").exists() {
        "Go"
    } else if canonical_root.join("Package.swift").exists() {
        "Swift (SPM)"
    } else {
        "Local Project"
    };

    let mut files = Vec::new();
    scan_dir_recursive(&canonical_root, &canonical_root, 0, &mut files);

    Json(ProjectScanResponse {
        status: "ok".to_string(),
        root_path: canonical_root.display().to_string(),
        project_name,
        project_type: project_type.to_string(),
        total_files: files.len(),
        files,
    })
    .into_response()
}

fn scan_dir_recursive(root: &Path, current: &Path, depth: usize, out: &mut Vec<ProjectFileItem>) {
    if depth > 5 || out.len() >= 250 {
        return;
    }

    let entries = match std::fs::read_dir(current) {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut sorted_entries = Vec::new();
    for entry in entries.flatten() {
        sorted_entries.push(entry);
    }
    sorted_entries.sort_by_key(|e| e.file_name());

    for entry in sorted_entries {
        if out.len() >= 250 {
            break;
        }
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(_) => continue,
        };

        let file_name = entry.file_name().to_string_lossy().to_string();

        if file_name.starts_with('.')
            || file_name == "target"
            || file_name == "node_modules"
            || file_name == "dist"
            || file_name == "build"
            || file_name == "out"
            || file_name == "__pycache__"
            || file_name == "vendor"
        {
            continue;
        }

        let path = entry.path();
        if file_type.is_dir() {
            scan_dir_recursive(root, &path, depth + 1, out);
        } else if file_type.is_file() {
            let rel_path = match path.strip_prefix(root) {
                Ok(p) => p.to_string_lossy().to_string(),
                Err(_) => continue,
            };

            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();

            let is_code = matches!(
                ext.as_str(),
                "rs" | "py" | "js" | "ts" | "tsx" | "jsx" | "go" | "c" | "cpp" | "h" | "hpp"
                    | "swift" | "java" | "kt" | "rb" | "php" | "sh" | "zsh" | "html" | "css"
                    | "json" | "yaml" | "yml" | "toml" | "md" | "sql"
            );

            let size_bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);

            out.push(ProjectFileItem {
                relative_path: rel_path,
                file_name,
                size_bytes,
                extension: ext,
                is_code,
            });
        }
    }
}

async fn handle_project_file(
    Json(req): Json<ProjectFileRequest>,
) -> Response {
    let root = Path::new(&req.root_path);
    let canonical_root = match root.canonicalize() {
        Ok(c) => c,
        Err(_) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "Invalid project root path." })),
            )
                .into_response();
        }
    };

    let target = canonical_root.join(&req.file_path);
    let canonical_target = match target.canonicalize() {
        Ok(c) => c,
        Err(_) => {
            return (
                axum::http::StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": "File not found." })),
            )
                .into_response();
        }
    };

    if !canonical_target.starts_with(&canonical_root) {
        return (
            axum::http::StatusCode::FORBIDDEN,
            Json(serde_json::json!({ "error": "Access outside project directory forbidden." })),
        )
            .into_response();
    }

    if !canonical_target.is_file() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "Requested path is not a file." })),
        )
            .into_response();
    }

    let metadata = match std::fs::metadata(&canonical_target) {
        Ok(m) => m,
        Err(e) => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("Failed to read metadata: {}", e) })),
            )
                .into_response();
        }
    };

    let size_bytes = metadata.len();
    if size_bytes > 2 * 1024 * 1024 {
        return (
            axum::http::StatusCode::PAYLOAD_TOO_LARGE,
            Json(serde_json::json!({ "error": "File exceeds 2 MB text limit for project view." })),
        )
            .into_response();
    }

    let bytes = match std::fs::read(&canonical_target) {
        Ok(b) => b,
        Err(e) => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("Failed to read file: {}", e) })),
            )
                .into_response();
        }
    };

    let content = String::from_utf8_lossy(&bytes).to_string();
    let line_count = content.lines().count();

    Json(ProjectFileResponse {
        status: "ok".to_string(),
        file_path: req.file_path,
        content,
        size_bytes,
        line_count,
    })
    .into_response()
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
                            InferenceEngine::load(&path_clone, gpu_layers, use_mlock, kv_mode, ctx_size)
                        })
                        .await;

                        if let Ok(Ok(new_engine)) = load_res {
                            let mut inner = state.inner.write().await;
                            inner.model_name = new_path
                                .file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_else(|| req_trim.to_string());
                            inner.model_path = new_path;
                            inner.engine = new_engine;
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
                    if data.starts_with("data:") || (!data.starts_with('/') && !data.starts_with('~') && !data.starts_with('.')) {
                        Attachment::from_base64(&filename, &data)
                            .or_else(|_| Attachment::from_file(&data))
                    } else {
                        Attachment::from_file(&data)
                            .or_else(|_| Attachment::from_base64(&filename, &data))
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
    let req_id = format!("chatcmpl-{now}");

    let config = GenerationConfig {
        max_tokens: payload.max_tokens,
        temperature: payload.temperature,
        min_p: payload.min_p,
        top_p: payload.top_p,
        top_k: payload.top_k,
        use_ngram_speculative: payload.ngram_speculative,
        seed: payload.seed,
    };

    let (tx, mut rx) = unbounded_channel();
    let cancel = Arc::new(AtomicBool::new(false));

    // Register active cancel token so /v1/chat/stop can immediately abort
    {
        let mut active = state.active_cancel.lock().await;
        *active = Some(cancel.clone());
    }

    let prompt_clone = prompt.clone();
    let cancel_clone = cancel.clone();
    let engine_clone = engine.clone();

    tokio::task::spawn_blocking(move || {
        let tx_err = tx.clone();
        if let Err(e) = engine_clone.stream_generate_with_config(&prompt_clone, &config, cancel_clone, tx) {
            let _ = tx_err.send(StreamEvent::Error(e.to_string()));
        }
    });

    if payload.stream {
        // Stream SSE tokens
        let model_name_for_stream = active_model_name.clone();
        let cancel_guard = CancelOnDrop(cancel.clone());
        let stream = async_stream::stream! {
            let _guard = cancel_guard;
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
                    StreamEvent::Stats {
                        ttft_ms,
                        tokens_per_sec,
                        total_tokens,
                        prompt_tokens,
                        context_used,
                        context_capacity,
                        prefix_tokens_reused,
                        prefix_cache_hit,
                        ..
                    } => {
                        let stats_json = serde_json::json!({
                            "id": req_id.clone(),
                            "object": "chat.completion.chunk",
                            "created": now,
                            "model": model_name_for_stream.clone(),
                            "choices": [],
                            "stats": {
                                "ttft_ms": ttft_ms,
                                "tokens_per_sec": tokens_per_sec,
                                "total_tokens": total_tokens,
                                "prompt_tokens": prompt_tokens,
                                "context_used": context_used,
                                "context_capacity": context_capacity,
                                "prefix_tokens_reused": prefix_tokens_reused,
                                "prefix_cache_hit": prefix_cache_hit,
                            }
                        });
                        yield Ok(Event::default().data(stats_json.to_string()));
                    }
                    StreamEvent::Done => {
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

#[allow(clippy::too_many_arguments)]
pub async fn run_server(
    engine: InferenceEngine,
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
        active_cancel: Arc::new(tokio::sync::Mutex::new(None)),
    };

    let app = create_router(state);

    println!("\n⚡ [NIRVANA CODE] Web UI & OpenAI-Compatible Local API Server");
    println!("   Platform:        Apple Silicon Metal 3 Unified LPDDR5");
    println!("   Model:           {model_name}");
    println!("   Web Interface:   http://{host}:{port}");
    println!("   Chat API URL:    http://{host}:{port}/v1/chat/completions");
    println!("   Models API URL:  http://{host}:{port}/v1/models");
    println!("   Switch API URL:  POST http://{host}:{port}/v1/models/load");

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

    let addr = format!("{host}:{port}");
    let tcp_listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(tcp_listener, app).await?;

    Ok(())
}
