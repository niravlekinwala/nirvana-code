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
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::unbounded_channel;
use tower_http::cors::CorsLayer;

use crate::engine::{GenerationConfig, ModelEngine, StreamEvent};

#[derive(Clone)]
pub struct ServerState {
    pub engine: Arc<ModelEngine>,
    pub model_name: String,
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
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
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
        .route("/v1/chat/completions", post(handle_chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn handle_index() -> Html<&'static str> {
    Html(include_str!("web/index.html"))
}

async fn handle_models(State(state): State<ServerState>) -> Json<ModelListResponse> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Json(ModelListResponse {
        object: "list",
        data: vec![
            ModelCard {
                id: "nirvana-code".to_string(),
                object: "model",
                created: now,
                owned_by: "nirvana",
            },
            ModelCard {
                id: state.model_name.clone(),
                object: "model",
                created: now,
                owned_by: "nirvana",
            },
        ],
    })
}

fn format_messages_to_prompt(messages: &[ChatMessageDto]) -> String {
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

    for msg in messages {
        if msg.role.to_lowercase() != "system" {
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
    let prompt = format_messages_to_prompt(&payload.messages);
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

    let engine = state.engine.clone();
    let prompt_clone = prompt.clone();
    let cancel_clone = cancel.clone();

    tokio::task::spawn_blocking(move || {
        let _ = engine.stream_generate_with_config(&prompt_clone, &config, cancel_clone, tx);
    });

    if payload.stream {
        // Stream SSE tokens
        let stream = async_stream::stream! {
            // First chunk with role
            let initial_chunk = ChatCompletionChunk {
                id: req_id.clone(),
                object: "chat.completion.chunk",
                created: now,
                model: state.model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: Some("assistant".to_string()),
                        content: None,
                    },
                    finish_reason: None,
                }],
            };
            if let Ok(data) = serde_json::to_string(&initial_chunk) {
                yield Ok::<Event, Infallible>(Event::default().data(data));
            }

            while let Some(event) = rx.recv().await {
                match event {
                    StreamEvent::Token(tok) => {
                        let chunk = ChatCompletionChunk {
                            id: req_id.clone(),
                            object: "chat.completion.chunk",
                            created: now,
                            model: state.model_name.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: Some(tok),
                                },
                                finish_reason: None,
                            }],
                        };
                        if let Ok(data) = serde_json::to_string(&chunk) {
                            yield Ok::<Event, Infallible>(Event::default().data(data));
                        }
                    }
                    StreamEvent::Stats { .. } => {}
                    StreamEvent::Done => {
                        let final_chunk = ChatCompletionChunk {
                            id: req_id.clone(),
                            object: "chat.completion.chunk",
                            created: now,
                            model: state.model_name.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta { role: None, content: None },
                                finish_reason: Some("stop"),
                            }],
                        };
                        if let Ok(data) = serde_json::to_string(&final_chunk) {
                            yield Ok::<Event, Infallible>(Event::default().data(data));
                        }
                        yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
                        break;
                    }
                    StreamEvent::Error(err) => {
                        yield Ok::<Event, Infallible>(Event::default().data(format!("Error: {}", err)));
                        break;
                    }
                }
            }
        };

        Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
    } else {
        // Collect full response
        let mut full_text = String::new();
        let mut total_toks = 0;

        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(tok) => full_text.push_str(&tok),
                StreamEvent::Stats { total_tokens, .. } => total_toks = total_tokens,
                StreamEvent::Done => break,
                StreamEvent::Error(err) => {
                    return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, err).into_response();
                }
            }
        }

        let resp = ChatCompletionResponse {
            id: req_id,
            object: "chat.completion",
            created: now,
            model: state.model_name,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessageDto {
                    role: "assistant".to_string(),
                    content: full_text,
                },
                finish_reason: "stop",
            }],
            usage: UsageInfo {
                prompt_tokens: 0,
                completion_tokens: total_toks,
                total_tokens: total_toks,
            },
        };

        Json(resp).into_response()
    }
}

pub async fn run_server(
    engine: Arc<ModelEngine>,
    model_name: String,
    host: &str,
    port: u16,
    socket_path: Option<PathBuf>,
) -> Result<()> {
    let state = ServerState {
        engine,
        model_name: model_name.clone(),
    };

    let app = create_router(state);

    println!("\n⚡ [NIRVANA CODE] Web UI & OpenAI-Compatible Local API Server");
    println!("   Platform:        Apple Silicon Metal 3 Unified LPDDR5");
    println!("   Model:           {}", model_name);
    println!("   Web Interface:   http://{}:{}", host, port);
    println!("   Chat API URL:    http://{}:{}/v1/chat/completions", host, port);
    println!("   Models API URL:  http://{}:{}/v1/models", host, port);

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

    println!("\n   Ready for VS Code Continue.dev, Cursor, Neovim, Zed, and OpenAI SDKs!");
    println!("   Press Ctrl+C to stop.\n");

    let addr = format!("{}:{}", host, port);
    let tcp_listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(tcp_listener, app).await?;

    Ok(())
}
