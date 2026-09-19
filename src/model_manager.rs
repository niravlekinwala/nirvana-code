use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ModelMeta {
    pub id: &'static str,
    pub name: &'static str,
    pub filename: &'static str,
    pub url: &'static str,
    pub size_gb: f32,
    pub category: &'static str,
    pub speed_m2_pro: &'static str,
    pub description: &'static str,
}

pub const MODEL_CATALOG: &[ModelMeta] = &[
    // 27B & 35B Heavyweight Models (From 16GB Guide)
    ModelMeta {
        id: "qwen-3.8-27b",
        name: "Qwen 3.8 27B (Atomic Dynamic IQ3_S)",
        filename: "Qwen3.8-27B-AD-IQ3_S.gguf",
        url: "https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-AD-IQ3_S.gguf",
        size_gb: 13.8,
        category: "Flagship 27B (16GB RAM Fit)",
        speed_m2_pro: "~20-25 tok/s",
        description: "Dense multimodal 27B with vision & thinking, quantized to fit inside 16GB Unified RAM with 8K context.",
    },
    ModelMeta {
        id: "qwen-3.8-27b-xxs",
        name: "Qwen 3.8 27B (IQ3_S / IQ3_XXS)",
        filename: "Qwen3.8-27B-AD-IQ3_S-IQ3_XXS.gguf",
        url: "https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-AD-IQ3_S-IQ3_XXS.gguf",
        size_gb: 13.0,
        category: "Flagship 27B (Extra Headroom)",
        speed_m2_pro: "~22-26 tok/s",
        description: "Slightly leaner 27B quant leaving 3GB free memory headroom on 16GB Macs for larger contexts.",
    },
    ModelMeta {
        id: "ornith-35b",
        name: "Ornith 1.5 35B-A3B (MoE)",
        filename: "Ornith-1.5-35B-A3B-AD-IQ3_XXS-IQ2_S.gguf",
        url: "https://huggingface.co/AtomicChat/Ornith-1.5-35B-A3B-GGUF/resolve/main/Ornith-1.5-35B-A3B-AD-IQ3_XXS-IQ2_S.gguf",
        size_gb: 13.7,
        category: "Agentic MoE (3B Active Params)",
        speed_m2_pro: "~35-45 tok/s",
        description: "35B Mixture-of-Experts activating only 3B parameters per token for high-speed agentic coding.",
    },

    // 7B - 12B Mid-Tier Champions (Best for 16GB RAM + 32K Context)
    ModelMeta {
        id: "qwen-3.5-9b",
        name: "Qwen 3.5 9B (Q6_K)",
        filename: "qwen35-9b-Q6_K.gguf",
        url: "https://huggingface.co/AtomicChat/Qwen3.5-9B-GGUF/resolve/main/qwen35-9b-Q6_K.gguf",
        size_gb: 7.4,
        category: "Best Overall for 16GB Unified RAM",
        speed_m2_pro: "~55-65 tok/s",
        description: "Ranked #1 for 16GB RAM machines: outstanding capability with room for 32K context and OS apps.",
    },
    ModelMeta {
        id: "ornith-9b",
        name: "Ornith 1.5 9B (AD-Q8_0-Q6_K)",
        filename: "Ornith-1.5-9B-AD-Q8_0-Q6_K.gguf",
        url: "https://huggingface.co/AtomicChat/Ornith-1.5-9B-GGUF/resolve/main/Ornith-1.5-9B-AD-Q8_0-Q6_K.gguf",
        size_gb: 8.6,
        category: "Specialized Coding & Reasoning",
        speed_m2_pro: "~50-60 tok/s",
        description: "Specialized programming model with internal reasoning traces for complex bug fixing.",
    },
    ModelMeta {
        id: "lfm-8b",
        name: "LFM2.5 8B-A1B (Q6_K)",
        filename: "lfm25-8b-a1b-Q6_K.gguf",
        url: "https://huggingface.co/AtomicChat/LFM2.5-8B-A1B-GGUF/resolve/main/lfm25-8b-a1b-Q6_K.gguf",
        size_gb: 7.0,
        category: "Fast Assistant & Tool Use",
        speed_m2_pro: "~65-75 tok/s",
        description: "Highly responsive architecture for chained tool calling and on-device assistant workflows.",
    },

    // Compact Fast Models (1B - 3B)
    ModelMeta {
        id: "qwen-coder-3b",
        name: "Qwen 2.5 Coder 3B Instruct",
        filename: "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        url: "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        size_gb: 2.1,
        category: "Compact Coding & Prompts",
        speed_m2_pro: "~95 tok/s",
        description: "Compact model for everyday coding, debugging, and prompt structuring.",
    },
    ModelMeta {
        id: "qwen-coder-1.5b",
        name: "Qwen 2.5 Coder 1.5B Instruct",
        filename: "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        url: "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        size_gb: 1.0,
        category: "Ultra-Fast Coding & Drafting",
        speed_m2_pro: "~140 tok/s",
        description: "Blistering fast token generation, ideal for instant code generation and speculative drafting.",
    },
    ModelMeta {
        id: "qwen-1.5b",
        name: "Qwen 2.5 1.5B Instruct",
        filename: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        size_gb: 1.0,
        category: "Fast Assistant & Drafting",
        speed_m2_pro: "~140 tok/s",
        description: "Compact general instruction & coding model with ultra-fast decode speed.",
    },
    ModelMeta {
        id: "deepseek-r1-1.5b",
        name: "DeepSeek R1 Distill Qwen 1.5B",
        filename: "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        url: "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        size_gb: 1.1,
        category: "Chain-of-Thought Reasoning",
        speed_m2_pro: "~120 tok/s",
        description: "Offline reasoning model that shows internal thinking process (<think>) before answering.",
    },
    ModelMeta {
        id: "qwen-0.5b",
        name: "Qwen 2.5 0.5B Instruct (Speculative Draft)",
        filename: "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        size_gb: 0.4,
        category: "Ultra-Fast Speculative Draft",
        speed_m2_pro: "~190 tok/s",
        description: "Tiny 400MB model for ultra-low latency completion and speculative drafting with flagship models.",
    },
    ModelMeta {
        id: "qwen-coder-14b-iq3",
        name: "Qwen 2.5 Coder 14B (IQ3_M)",
        filename: "Qwen2.5-Coder-14B-Instruct-IQ3_M.gguf",
        url: "https://huggingface.co/bartowski/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-14B-Instruct-IQ3_M.gguf",
        size_gb: 6.8,
        category: "Flagship 14B (16GB RAM Fit)",
        speed_m2_pro: "~38-45 tok/s",
        description: "High-intelligence coding model quantized to 3-bit IQ3_M to fit inside 16GB Macs with 16K context.",
    },
    ModelMeta {
        id: "llama-3.2-3b",
        name: "Llama 3.2 3B Instruct",
        filename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        size_gb: 2.0,
        category: "General Assistant",
        speed_m2_pro: "~100 tok/s",
        description: "Meta's lightweight instruction model, sharp at following concise rules and formatting.",
    },
];

pub struct ModelManager;

impl ModelManager {
    pub fn default_dir() -> Result<PathBuf> {
        let home = dirs::home_dir().context("Could not determine user home directory")?;
        let dir = home.join(".nirvana").join("models");
        fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    pub fn secondary_dir() -> Option<PathBuf> {
        dirs::home_dir().map(|h| h.join(".promptcraft").join("models"))
    }

    pub fn lmstudio_dir() -> Option<PathBuf> {
        if let Some(home) = dirs::home_dir() {
            // 1. Check ~/.lmstudio/settings.json for explicit downloadsFolder
            let settings_file = home.join(".lmstudio").join("settings.json");
            if settings_file.exists() {
                if let Ok(content) = fs::read_to_string(&settings_file) {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
                        if let Some(df) = v.get("downloadsFolder").and_then(|s| s.as_str()) {
                            let p = PathBuf::from(df);
                            if p.exists() {
                                return Some(p);
                            }
                        }
                    }
                }
            }
            // 2. Default ~/.lmstudio/models
            let default_lm = home.join(".lmstudio").join("models");
            if default_lm.exists() {
                return Some(default_lm);
            }
        }
        None
    }

    /// Check if a path points to an Apple MLX model directory (contains config.json)
    pub fn is_mlx_model(path: &Path) -> bool {
        if path.is_dir() {
            path.join("config.json").exists()
        } else {
            false
        }
    }

    /// Check if a path points to a GGUF model file
    #[allow(dead_code)]
    pub fn is_gguf_model(path: &Path) -> bool {
        path.is_file()
            && path
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false)
    }

    /// Compute total size of all files inside an MLX directory
    pub fn compute_dir_size(dir: &Path) -> u64 {
        let mut total = 0;
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    if meta.is_file() {
                        total += meta.len();
                    }
                }
            }
        }
        total
    }

    pub fn list_installed() -> Vec<(PathBuf, String, u64)> {
        let mut list = Vec::new();
        let mut seen = std::collections::HashSet::new();

        // 1. Check primary ~/.nirvana/models
        if let Ok(dir) = Self::default_dir() {
            Self::scan_dir_recursive(&dir, 0, 4, &mut list, &mut seen);
        }

        // 2. Check LM Studio models folder (~/.lmstudio/models)
        if let Some(dir) = Self::lmstudio_dir() {
            Self::scan_dir_recursive(&dir, 0, 5, &mut list, &mut seen);
        }

        // 3. Check secondary ~/.promptcraft/models
        if let Some(dir) = Self::secondary_dir() {
            if dir.exists() {
                Self::scan_dir_recursive(&dir, 0, 4, &mut list, &mut seen);
            }
        }

        // 4. Check current working directory models/
        let local = PathBuf::from("models");
        if local.exists() {
            Self::scan_dir_recursive(&local, 0, 4, &mut list, &mut seen);
        }

        list
    }

    fn scan_dir_recursive(
        dir: &Path,
        depth: usize,
        max_depth: usize,
        list: &mut Vec<(PathBuf, String, u64)>,
        seen: &mut std::collections::HashSet<PathBuf>,
    ) {
        if depth > max_depth || !dir.exists() {
            return;
        }

        // 1. Check if current directory is an MLX model directory (has config.json)
        if dir.is_dir() && Self::is_mlx_model(dir) {
            let canon = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
            if !seen.contains(&canon) {
                seen.insert(canon);
                let folder_name = dir
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "mlx-model".to_string());
                let name = format!("{} [MLX]", folder_name);
                let size = Self::compute_dir_size(dir);
                list.push((dir.to_path_buf(), name, size));
            }
            return; // Do not scan inside the MLX model directory
        }

        // 2. Traverse directory contents
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    let filename = path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();

                    // Skip projection / adapter weights (e.g. mmproj-...) from standalone list
                    if filename.starts_with("mmproj") {
                        continue;
                    }

                    if path.extension().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("gguf")).unwrap_or(false) {
                        let canon = path.canonicalize().unwrap_or_else(|_| path.clone());
                        if !seen.contains(&canon) {
                            seen.insert(canon);
                            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                            list.push((path, filename, size));
                        }
                    }
                } else if path.is_dir() {
                    let dirname = path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    if !dirname.starts_with('.') {
                        Self::scan_dir_recursive(&path, depth + 1, max_depth, list, seen);
                    }
                }
            }
        }
    }

    pub fn resolve_model_path(explicit_path: Option<&Path>) -> Option<PathBuf> {
        // 1. Explicit CLI argument (path, catalog ID, substring, or index number)
        if let Some(path) = explicit_path {
            if path.exists() {
                return Some(path.to_path_buf());
            }
            let target = path.to_string_lossy();
            let installed = Self::list_installed();

            // 1a. 1-based number selection (e.g. /model 1, /model 2)
            if let Ok(idx) = target.trim().parse::<usize>() {
                if idx >= 1 && idx <= installed.len() {
                    return Some(installed[idx - 1].0.clone());
                }
            }

            // 1b. Exact or substring match
            for (p, name, _) in &installed {
                if name == target.as_ref() || name.to_lowercase().contains(&target.to_lowercase()) {
                    return Some(p.clone());
                }
            }

            // 1c. Catalog ID match
            if let Some(meta) = MODEL_CATALOG.iter().find(|m| m.id == target.as_ref()) {
                for (p, name, _) in &installed {
                    if name == &meta.filename {
                        return Some(p.clone());
                    }
                }
            }
        }

        // 2. Environment variable
        if let Ok(env_path) = std::env::var("NIRVANA_MODEL") {
            let p = PathBuf::from(env_path);
            if p.exists() {
                return Some(p);
            }
        }

        // 3. Current directory models/
        if let Ok(entries) = fs::read_dir("models") {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                    return Some(path);
                }
            }
        }

        // 4. Smart default priority among installed models:
        // Prioritize responsive everyday models (1.5B, 3B, 9B), then 27B flagships
        let installed = Self::list_installed();
        let preferred_ids = [
            "qwen-1.5b",
            "qwen-coder-1.5b",
            "qwen-coder-3b",
            "qwen-3.5-9b",
            "lfm-8b",
            "deepseek-r1-1.5b",
            "llama-3.2-3b",
            "qwen-3.8-27b",
            "ornith-35b",
        ];

        for id in preferred_ids {
            if let Some(meta) = MODEL_CATALOG.iter().find(|m| m.id == id) {
                if let Some((path, _, _)) = installed.iter().find(|(_, name, _)| name == meta.filename) {
                    return Some(path.clone());
                }
            }
        }

        // Fallback: any installed model
        installed.into_iter().next().map(|(p, _, _)| p)
    }

    pub async fn download_target(target: &str) -> Result<PathBuf> {
        // If it's a full URL
        if target.starts_with("http://") || target.starts_with("https://") {
            let filename = target
                .split('/')
                .last()
                .unwrap_or("custom_model.gguf")
                .to_string();
            return Self::download_file(&filename, target).await;
        }

        // If it's a catalog ID or filename
        let meta = MODEL_CATALOG
            .iter()
            .find(|m| {
                m.id == target
                    || m.filename == target
                    || m.name.to_lowercase().contains(&target.to_lowercase())
            })
            .context(format!(
                "Unknown model target '{}'. Run 'nirvana-code models' to view available models.",
                target
            ))?;

        Self::download_file(meta.filename, meta.url).await
    }

    pub async fn download_file(filename: &str, url: &str) -> Result<PathBuf> {
        let dir = Self::default_dir()?;
        let target_path = dir.join(filename);
        if target_path.exists() {
            println!("Model already exists at {}", target_path.display());
            return Ok(target_path);
        }

        let temp_path = dir.join(format!("{}.download", filename));

        println!("⚡ [NIRVANA CODE] Downloading Silicon-Optimized Model for Apple Silicon:");
        println!("   Filename: {}", filename);
        println!("   URL:      {}", url);
        println!("   Target:   {}\n", target_path.display());

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3600)) // 1 hour timeout for large models
            .build()?;

        let res = client
            .get(url)
            .send()
            .await
            .context("Failed to initiate download stream")?;
        let status = res.status();
        if !status.is_success() {
            bail!("Download request failed with HTTP status: {}", status);
        }

        let total_size = res.content_length().unwrap_or(0);
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?
                .progress_chars("#>-"),
        );

        let mut file = File::create(&temp_path)?;
        let mut stream = res.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let data = chunk.context("Error while downloading stream chunk")?;
            file.write_all(&data)?;
            pb.inc(data.len() as u64);
        }

        pb.finish_with_message("Download complete!");
        drop(file);

        fs::rename(&temp_path, &target_path)?;
        println!("\n✔ Model successfully cached at: {}", target_path.display());

        Ok(target_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_mlx_model_detection() {
        let temp_dir = std::env::temp_dir().join("nirvana_test_mlx");
        let _ = fs::create_dir_all(&temp_dir);
        let _ = fs::write(temp_dir.join("config.json"), "{}");
        assert!(ModelManager::is_mlx_model(&temp_dir));

        let non_mlx = std::env::temp_dir().join("nirvana_test_non_mlx");
        let _ = fs::create_dir_all(&non_mlx);
        assert!(!ModelManager::is_mlx_model(&non_mlx));

        let _ = fs::remove_dir_all(&temp_dir);
        let _ = fs::remove_dir_all(&non_mlx);
    }

    #[test]
    fn test_list_installed_includes_lmstudio() {
        let installed = ModelManager::list_installed();
        // User has LM Studio models installed, verify discovery
        if let Some(lm_dir) = ModelManager::lmstudio_dir() {
            if lm_dir.exists() {
                let has_lmstudio = installed.iter().any(|(p, _, _)| p.starts_with(&lm_dir));
                assert!(has_lmstudio, "Expected installed models to include LM Studio models");
            }
        }
    }
}
