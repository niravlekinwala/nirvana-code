use anyhow::{bail, Context, Result};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::UnboundedSender;

use crate::chat::ChatMessage;
use crate::engine::{GenerationConfig, StreamEvent};

pub struct MlxEngine {
    pub model_path: PathBuf,
    pub model_name: String,
    pub port: u16,
    /// From the model's `config.json`; 0 if unknown.
    pub n_layers: u32,
    /// Context size the caller asked for; mlx_lm manages its own cache, this is
    /// only reported back in stats.
    pub n_ctx: u32,
    process: Arc<Mutex<Option<Child>>>,
    client: reqwest::Client,
}

impl MlxEngine {
    /// Detect or resolve the MLX command and arguments
    pub fn resolve_mlx_command() -> Result<(String, Vec<String>)> {
        // 1. Direct `mlx_lm` in PATH
        if Command::new("mlx_lm")
            .arg("--help")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        {
            return Ok(("mlx_lm".to_string(), vec![]));
        }

        // 2. Check Anaconda/Miniconda paths
        if let Some(home) = dirs::home_dir() {
            let candidates = [
                home.join("Softwares/anaconda3/bin/mlx_lm"),
                home.join("anaconda3/bin/mlx_lm"),
                home.join("miniconda3/bin/mlx_lm"),
                home.join(".miniforge3/bin/mlx_lm"),
                PathBuf::from("/opt/homebrew/bin/mlx_lm"),
                PathBuf::from("/usr/local/bin/mlx_lm"),
            ];

            for cand in candidates {
                if cand.exists() {
                    let cand_str = cand.to_string_lossy().to_string();
                    if Command::new(&cand_str)
                        .arg("--help")
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .status()
                        .map(|s| s.success())
                        .unwrap_or(false)
                    {
                        return Ok((cand_str, vec![]));
                    }
                }
            }
        }

        // 3. Fallback: `python3 -m mlx_lm`
        if Command::new("python3")
            .args(["-m", "mlx_lm", "--help"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        {
            return Ok(("python3".to_string(), vec!["-m".to_string(), "mlx_lm".to_string()]));
        }

        bail!("Could not find Apple MLX ('mlx_lm'). Please install it using: pip install mlx-lm")
    }

    /// Allocate an available ephemeral local port
    fn allocate_local_port() -> Result<u16> {
        let listener = TcpListener::bind("127.0.0.1:0")
            .context("Failed to bind ephemeral local TCP port for MLX server")?;
        let port = listener.local_addr()?.port();
        drop(listener);
        Ok(port)
    }

    /// Poll until the MLX server HTTP endpoint responds 200 OK
    fn wait_for_server_ready(
        port: u16,
        timeout: Duration,
        process: &Arc<Mutex<Option<Child>>>,
    ) -> Result<()> {
        let start_wait = Instant::now();
        let addr_str = format!("127.0.0.1:{port}");
        let sock_addr: SocketAddr = addr_str
            .parse()
            .context("Failed to parse local server address")?;

        while start_wait.elapsed() < timeout {
            // Check if child process died
            if let Ok(mut lock) = process.lock() {
                if let Some(ref mut ch) = *lock {
                    if let Ok(Some(status)) = ch.try_wait() {
                        let mut err_detail = String::new();
                        if let Some(ref mut stderr) = ch.stderr {
                            let _ = stderr.read_to_string(&mut err_detail);
                        }
                        bail!(
                            "MLX server exited prematurely with code {:?}. Details: {}",
                            status.code(),
                            err_detail.trim()
                        );
                    }
                }
            }

            // Try TCP connection and HTTP GET request
            if let Ok(mut stream) = TcpStream::connect_timeout(&sock_addr, Duration::from_millis(300)) {
                let req = format!(
                    "GET /v1/models HTTP/1.1\r\nHost: {addr_str}\r\nConnection: close\r\n\r\n"
                );
                if stream.write_all(req.as_bytes()).is_ok() {
                    let mut resp_buf = [0u8; 256];
                    if let Ok(n) = stream.read(&mut resp_buf) {
                        if n > 0 {
                            let resp_str = String::from_utf8_lossy(&resp_buf[..n]);
                            if resp_str.contains("200 OK") || resp_str.contains("HTTP/1.1 200") {
                                return Ok(());
                            }
                        }
                    }
                }
            }

            std::thread::sleep(Duration::from_millis(100));
        }

        bail!(
            "Timed out waiting for MLX model server to initialize on port {port}"
        );
    }

    /// Load and launch the Apple MLX model server as an optimized background process
    pub fn load(model_path: &Path, n_ctx: u32) -> Result<Self> {
        let (cmd_bin, cmd_base_args) = Self::resolve_mlx_command()?;
        let n_layers = std::fs::read_to_string(model_path.join("config.json"))
            .ok()
            .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok())
            .and_then(|v| v.get("num_hidden_layers").and_then(|n| n.as_u64()))
            .unwrap_or(0) as u32;
        let port = Self::allocate_local_port()?;

        let model_name = model_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "mlx-model".to_string());

        let mut args = cmd_base_args;
        args.push("server".to_string());
        args.push("--model".to_string());
        args.push(model_path.to_string_lossy().to_string());
        args.push("--port".to_string());
        args.push(port.to_string());
        args.push("--log-level".to_string());
        args.push("ERROR".to_string());

        // Spawn child process with suppressed outputs to protect TUI terminal state
        let mut cmd = Command::new(&cmd_bin);
        cmd.args(&args)
            .stdout(Stdio::null())
            .stderr(Stdio::piped());

        let child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn MLX server process using {cmd_bin}"))?;

        let process = Arc::new(Mutex::new(Some(child)));

        // Wait for server ready (up to 25s for large quants)
        if let Err(e) = Self::wait_for_server_ready(port, Duration::from_secs(25), &process) {
            if let Ok(mut lock) = process.lock() {
                if let Some(mut ch) = lock.take() {
                    let _ = ch.kill();
                    let _ = ch.wait();
                }
            }
            return Err(e);
        }

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()?;

        Ok(Self {
            model_path: model_path.to_path_buf(),
            model_name,
            port,
            n_layers,
            n_ctx,
            process,
            client,
        })
    }

    /// Explicitly clear cache / state
    pub fn clear_cache(&self) {
        // MLX automatically manages prompt KV cache per request prefix
    }

    /// Parse raw ChatML or text prompt into structured chat messages
    pub fn parse_prompt_to_messages(prompt: &str) -> Vec<serde_json::Value> {
        let mut messages = Vec::new();
        let marker = "<|im_start|>";
        let end_marker = "<|im_end|>";

        if !prompt.contains(marker) {
            messages.push(serde_json::json!({
                "role": "user",
                "content": prompt.trim()
            }));
            return messages;
        }

        let mut remainder = prompt;
        while let Some(start_idx) = remainder.find(marker) {
            remainder = &remainder[start_idx + marker.len()..];
            if let Some(end_idx) = remainder.find(end_marker) {
                let block = &remainder[..end_idx];
                remainder = &remainder[end_idx + end_marker.len()..];

                if let Some(newline_idx) = block.find('\n') {
                    let role = block[..newline_idx].trim().to_lowercase();
                    let content = block[newline_idx + 1..].trim().to_string();
                    if !role.is_empty() && !content.is_empty() {
                        messages.push(serde_json::json!({
                            "role": role,
                            "content": content
                        }));
                    }
                }
            } else {
                // Trailing turn header without end marker (e.g. <|im_start|>assistant\n)
                break;
            }
        }

        if messages.is_empty() {
            messages.push(serde_json::json!({
                "role": "user",
                "content": prompt.trim()
            }));
        }

        messages
    }

    /// Raw-prompt entry point: recovers chat turns from ChatML markup. Prefer
    /// [`Self::stream_chat`].
    pub fn stream_generate_with_config(
        &self,
        prompt: &str,
        config: &GenerationConfig,
        cancel_token: Arc<AtomicBool>,
        tx: UnboundedSender<StreamEvent>,
    ) -> Result<()> {
        let messages = Self::parse_prompt_to_messages(prompt);
        self.stream_messages(messages, config, cancel_token, tx)
    }

    /// Stream a conversation; mlx_lm applies the model's own chat template.
    pub fn stream_chat(
        &self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        cancel_token: Arc<AtomicBool>,
        tx: UnboundedSender<StreamEvent>,
    ) -> Result<()> {
        let messages = messages
            .iter()
            .map(|m| serde_json::json!({ "role": m.role, "content": m.content }))
            .collect();
        self.stream_messages(messages, config, cancel_token, tx)
    }

    fn stream_messages(
        &self,
        messages: Vec<serde_json::Value>,
        config: &GenerationConfig,
        cancel_token: Arc<AtomicBool>,
        tx: UnboundedSender<StreamEvent>,
    ) -> Result<()> {
        let url = format!("http://127.0.0.1:{}/v1/chat/completions", self.port);

        let body = serde_json::json!({
            "model": self.model_path.to_string_lossy(),
            "messages": messages,
            "stream": true,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "repetition_penalty": 1.15,
            "repetition_context_size": 64,
            "stop": ["<|im_end|>", "<|endoftext|>", "</s>"],
            "stream_options": {"include_usage": true}
        });

        let client = self.client.clone();
        let n_ctx = self.n_ctx;
        let cancel_clone = cancel_token.clone();
        let tx_clone = tx.clone();

        let runtime = tokio::runtime::Handle::current();
        runtime.block_on(async move {
            let res = match client.post(&url).json(&body).send().await {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx_clone.send(StreamEvent::Error(format!("MLX connection failed: {e}")));
                    return;
                }
            };

            if !res.status().is_success() {
                let err_text = res.text().await.unwrap_or_else(|_| "Unknown HTTP error".to_string());
                let _ = tx_clone.send(StreamEvent::Error(format!("MLX server error: {err_text}")));
                return;
            }

            let mut mut_res = res;
            let mut buffer = String::new();
            let start_time = Instant::now();
            let mut ttft_ms = 0;
            let mut token_count = 0;
            let mut cached_tokens = 0;
            let mut prompt_tokens = 0usize;
            let mut in_reasoning = false;
            let mut recent_chars: Vec<char> = Vec::new();

            loop {
                if cancel_clone.load(Ordering::Relaxed) {
                    break;
                }

                let chunk = match mut_res.chunk().await {
                    Ok(Some(c)) => c,
                    Ok(None) => break,
                    Err(e) => {
                        let _ = tx_clone.send(StreamEvent::Error(format!("MLX stream read error: {e}")));
                        return;
                    }
                };

                buffer.push_str(&String::from_utf8_lossy(&chunk));

                let mut loop_detected = false;

                while let Some(nl_idx) = buffer.find('\n') {
                    if cancel_clone.load(Ordering::Relaxed) {
                        break;
                    }

                    let line = buffer[..nl_idx].trim().to_string();
                    buffer = buffer[nl_idx + 1..].to_string();

                    if line.is_empty() || line.starts_with(':') {
                        continue;
                    }

                    if line == "data: [DONE]" {
                        break;
                    }

                    if let Some(json_str) = line.strip_prefix("data: ") {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
                            // Extract cached tokens usage if present
                            if let Some(usage) = v.get("usage") {
                                if let Some(pt) = usage.get("prompt_tokens").and_then(|c| c.as_u64()) {
                                    prompt_tokens = pt as usize;
                                }
                                if let Some(details) = usage.get("prompt_tokens_details") {
                                    if let Some(ct) = details.get("cached_tokens").and_then(|c| c.as_u64()) {
                                        cached_tokens = ct as usize;
                                    }
                                }
                            }

                            // Extract delta token
                            if let Some(choices) = v.get("choices").and_then(|c| c.as_array()) {
                                if let Some(first) = choices.first() {
                                    if let Some(delta) = first.get("delta") {
                                        // Reasoning content (DeepSeek-R1 etc.)
                                        if let Some(reasoning) = delta.get("reasoning").and_then(|r| r.as_str()) {
                                            if !reasoning.is_empty() {
                                                if token_count == 0 {
                                                    ttft_ms = start_time.elapsed().as_millis();
                                                }
                                                token_count += 1;
                                                if !in_reasoning {
                                                    in_reasoning = true;
                                                    let _ = tx_clone.send(StreamEvent::Token("<think>\n".to_string()));
                                                }
                                                let _ = tx_clone.send(StreamEvent::Token(reasoning.to_string()));
                                            }
                                        }

                                        // Regular content
                                        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                                            if !content.is_empty() {
                                                if token_count == 0 {
                                                    ttft_ms = start_time.elapsed().as_millis();
                                                }
                                                token_count += 1;

                                                if in_reasoning {
                                                    in_reasoning = false;
                                                    let _ = tx_clone.send(StreamEvent::Token("\n</think>\n\n".to_string()));
                                                }

                                                // Repetition loop detector (character-based, UTF-8 safe)
                                                recent_chars.extend(content.chars());
                                                if recent_chars.len() > 300 {
                                                    let excess = recent_chars.len() - 300;
                                                    recent_chars.drain(..excess);
                                                }
                                                let rlen = recent_chars.len();
                                                if rlen >= 48 {
                                                    for pat_len in 6..=30 {
                                                        if rlen >= pat_len * 4 {
                                                            let tail = &recent_chars[rlen - pat_len..];
                                                            let prev1 = &recent_chars[rlen - pat_len * 2..rlen - pat_len];
                                                            let prev2 = &recent_chars[rlen - pat_len * 3..rlen - pat_len * 2];
                                                            let prev3 = &recent_chars[rlen - pat_len * 4..rlen - pat_len * 3];
                                                            if tail == prev1 && prev1 == prev2 && prev2 == prev3 {
                                                                loop_detected = true;
                                                                break;
                                                            }
                                                        }
                                                    }
                                                }

                                                let _ = tx_clone.send(StreamEvent::Token(content.to_string()));

                                                if loop_detected {
                                                    let _ = tx_clone.send(StreamEvent::Token("\n\n⚠️ *[Generation halted: repetition degeneration loop detected]*".to_string()));
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if loop_detected || cancel_clone.load(Ordering::Relaxed) {
                    break;
                }
            }

            if in_reasoning {
                let _ = tx_clone.send(StreamEvent::Token("\n</think>\n\n".to_string()));
            }

            let total_secs = start_time.elapsed().as_secs_f64();
            let tps = if total_secs > 0.0 {
                token_count as f64 / total_secs
            } else {
                0.0
            };

            let _ = tx_clone.send(StreamEvent::Stats {
                ttft_ms,
                tokens_per_sec: tps,
                total_tokens: token_count,
                prompt_tokens,
                context_used: token_count + prompt_tokens,
                context_capacity: n_ctx,
                prefix_tokens_reused: cached_tokens,
                prefix_cache_hit: cached_tokens > 0,
                kv_type: "MLX (managed by mlx_lm)".to_string(),
                mlock_active: false,
            });
            let _ = tx_clone.send(StreamEvent::Done);
        });

        Ok(())
    }
}

impl Drop for MlxEngine {
    fn drop(&mut self) {
        if let Ok(mut lock) = self.process.lock() {
            if let Some(mut child) = lock.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_prompt() {
        let prompt = "Hello from Apple Silicon!";
        let msgs = MlxEngine::parse_prompt_to_messages(prompt);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "Hello from Apple Silicon!");
    }

    #[test]
    fn test_parse_chatml_prompt() {
        let prompt = "<|im_start|>system\nYou are Nirvana Code.<|im_end|>\n<|im_start|>user\nWrite a queue.<|im_end|>\n<|im_start|>assistant\n";
        let msgs = MlxEngine::parse_prompt_to_messages(prompt);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "You are Nirvana Code.");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "Write a queue.");
    }

    #[test]
    fn test_utf8_char_repetition_detector_no_panic() {
        // Multi-byte chars: curly apostrophe ’ (\u{2019}, 3 bytes), emoji 🚀 (4 bytes), umlaut ä (2 bytes)
        let sample = "Let’s build 🚀 fast with MLX! Let’s build 🚀 fast with MLX! Let’s build 🚀 fast with MLX! Let’s build 🚀 fast with MLX! ";
        let mut recent_chars: Vec<char> = Vec::new();
        let mut loop_detected = false;

        for ch in sample.chars() {
            recent_chars.push(ch);
            if recent_chars.len() > 300 {
                let excess = recent_chars.len() - 300;
                recent_chars.drain(..excess);
            }
            let rlen = recent_chars.len();
            if rlen >= 48 {
                for pat_len in 6..=30 {
                    if rlen >= pat_len * 4 {
                        let tail = &recent_chars[rlen - pat_len..];
                        let prev1 = &recent_chars[rlen - pat_len * 2..rlen - pat_len];
                        let prev2 = &recent_chars[rlen - pat_len * 3..rlen - pat_len * 2];
                        let prev3 = &recent_chars[rlen - pat_len * 4..rlen - pat_len * 3];
                        if tail == prev1 && prev1 == prev2 && prev2 == prev3 {
                            loop_detected = true;
                            break;
                        }
                    }
                }
            }
        }
        assert!(loop_detected, "Repetition detector should detect 4 repeating patterns with multi-byte chars without panicking");
    }
}
