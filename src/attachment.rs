use anyhow::{bail, Context, Result};
use base64::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttachmentType {
    Pdf,
    Image,
    Document,
    Code,
    Text,
}

impl AttachmentType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Pdf => "PDF",
            Self::Image => "IMAGE",
            Self::Document => "DOCUMENT",
            Self::Code => "CODE",
            Self::Text => "TEXT",
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            Self::Pdf => "📄",
            Self::Image => "🖼️",
            Self::Document => "📝",
            Self::Code => "💻",
            Self::Text => "📃",
        }
    }

    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "pdf" => Self::Pdf,
            "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp" | "tiff" | "heic" | "svg" => {
                Self::Image
            }
            "docx" | "doc" | "rtf" | "odt" | "html" | "htm" => Self::Document,
            "rs" | "py" | "js" | "ts" | "tsx" | "jsx" | "go" | "c" | "cpp" | "h" | "hpp"
            | "cs" | "java" | "kt" | "swift" | "rb" | "php" | "sh" | "zsh" | "bash" | "sql"
            | "json" | "yaml" | "yml" | "toml" | "xml" | "csv" | "tsv" | "css" | "scss"
            | "vue" | "svelte" | "lua" | "zig" | "scala" | "r" | "dart" | "m" | "mm" => {
                Self::Code
            }
            _ => Self::Text,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attachment {
    pub filename: String,
    pub file_type: AttachmentType,
    pub size_bytes: u64,
    pub metadata_summary: String,
    pub extracted_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base64_data: Option<String>,
}

impl Attachment {
    pub fn from_file<P: AsRef<Path>>(input_path: P) -> Result<Self> {
        let path = resolve_path(input_path.as_ref())?;
        if !path.exists() {
            bail!("File does not exist: {}", path.display());
        }
        if !path.is_file() {
            bail!("Path is a directory, not a file: {}", path.display());
        }

        let metadata = fs::metadata(&path)
            .with_context(|| format!("Failed to read metadata for {}", path.display()))?;
        let size_bytes = metadata.len();

        let filename = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "attachment".to_string());

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        let file_type = AttachmentType::from_extension(ext);
        let size_str = format_size(size_bytes);

        let (metadata_summary, extracted_text) = match file_type {
            AttachmentType::Pdf => {
                let (pages, text) = extract_pdf(&path)?;
                let meta = format!("{} • {} page{}", size_str, pages, if pages == 1 { "" } else { "s" });
                (meta, text)
            }
            AttachmentType::Image => {
                let (dimensions, text) = extract_image(&path)?;
                let meta = if dimensions.is_empty() {
                    format!("{size_str} • Vision OCR")
                } else {
                    format!("{size_str} • {dimensions} • Vision OCR")
                };
                (meta, text)
            }
            AttachmentType::Document => {
                let text = extract_document(&path)?;
                let lines = text.lines().count();
                let meta = format!("{size_str} • {lines} lines");
                (meta, text)
            }
            AttachmentType::Code => {
                let text = extract_code_or_text(&path)?;
                let lines = text.lines().count();
                let meta = format!("{} • {} lines • {}", size_str, lines, ext.to_uppercase());
                (meta, text)
            }
            AttachmentType::Text => {
                let text = extract_code_or_text(&path)?;
                let lines = text.lines().count();
                let meta = format!("{size_str} • {lines} lines");
                (meta, text)
            }
        };

        Ok(Self {
            filename,
            file_type,
            size_bytes,
            metadata_summary,
            extracted_text,
            base64_data: None,
        })
    }

    pub fn from_base64(filename: &str, base64_payload: &str) -> Result<Self> {
        let clean_b64 = if let Some(idx) = base64_payload.find(";base64,") {
            &base64_payload[idx + 8..]
        } else {
            base64_payload.trim()
        };

        let sanitized_b64: String = clean_b64.chars().filter(|c| !c.is_whitespace()).collect();
        let decoded = BASE64_STANDARD
            .decode(&sanitized_b64)
            .with_context(|| "Failed to decode base64 payload")?;

        let temp_dir = std::env::temp_dir().join("nirvana_attachments");
        fs::create_dir_all(&temp_dir)?;

        let mut sanitized_name = filename
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '.' || c == '-' || c == '_' { c } else { '_' })
            .collect::<String>();

        // Infer extension if missing from filename
        if !sanitized_name.contains('.') && base64_payload.starts_with("data:") {
            if base64_payload.starts_with("data:application/pdf") {
                sanitized_name.push_str(".pdf");
            } else if base64_payload.starts_with("data:image/png") {
                sanitized_name.push_str(".png");
            } else if base64_payload.starts_with("data:image/jpeg") || base64_payload.starts_with("data:image/jpg") {
                sanitized_name.push_str(".jpg");
            } else if base64_payload.starts_with("data:image/webp") {
                sanitized_name.push_str(".webp");
            }
        }

        let temp_file_path = temp_dir.join(format!(
            "{}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            sanitized_name
        ));

        fs::write(&temp_file_path, &decoded)?;

        let res = Self::from_file(&temp_file_path);
        // Ensure cleanup of temporary file
        let _ = fs::remove_file(&temp_file_path);

        let mut att = res?;
        att.filename = filename.to_string();
        att.base64_data = Some(base64_payload.to_string());

        Ok(att)
    }

    pub fn format_prompt(&self, user_question: &str) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "[ATTACHED FILE: {} | Type: {} | {}]\n",
            self.filename,
            self.file_type.label(),
            self.metadata_summary
        ));
        out.push_str("--- BEGIN ATTACHED FILE CONTENT ---\n");
        out.push_str(&self.extracted_text);
        if !self.extracted_text.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("--- END ATTACHED FILE CONTENT ---\n\n");

        let trimmed = user_question.trim();
        if trimmed.is_empty() {
            out.push_str(&format!(
                "Please analyze the attached {} (`{}`) and provide a clear, comprehensive summary of its contents, structure, and key insights.",
                self.file_type.label().to_lowercase(),
                self.filename
            ));
        } else {
            out.push_str(trimmed);
        }

        out
    }
}

pub fn resolve_path(input: &Path) -> Result<PathBuf> {
    let raw = input.to_string_lossy().to_string();
    let trimmed = raw.trim();

    // Strip wrapping single or double quotes
    let unquoted = if (trimmed.starts_with('\'') && trimmed.ends_with('\''))
        || (trimmed.starts_with('"') && trimmed.ends_with('"'))
    {
        if trimmed.len() >= 2 {
            &trimmed[1..trimmed.len() - 1]
        } else {
            trimmed
        }
    } else {
        trimmed
    };

    // Unescape escaped spaces from terminal drag-and-drop
    let unescaped = unquoted.replace("\\ ", " ");

    if unescaped.starts_with("~/") || unescaped == "~" {
        if let Some(home) = dirs::home_dir() {
            if unescaped == "~" {
                return Ok(home);
            }
            return Ok(home.join(&unescaped[2..]));
        }
    }

    let p = PathBuf::from(unescaped);
    if p.is_relative() {
        if let Ok(cwd) = std::env::current_dir() {
            return Ok(cwd.join(p));
        }
    }
    Ok(p)
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Native helper compiled by build.rs (PDFKit + Vision). Empty when swiftc
/// was unavailable at build time.
const EXTRACT_HELPER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/nirvana-extract"));

/// Materialise the embedded helper under ~/.nirvana/bin once per binary
/// build (keyed by content length + version) and return its path.
#[allow(clippy::const_is_empty)] // empty only when build.rs found no swiftc
fn helper_path() -> Option<PathBuf> {
    if EXTRACT_HELPER.is_empty() {
        return None;
    }
    static PATH: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    PATH.get_or_init(|| {
        let dir = dirs::home_dir()?.join(".nirvana").join("bin");
        fs::create_dir_all(&dir).ok()?;
        let name = format!("nirvana-extract-{}-{}", env!("CARGO_PKG_VERSION"), EXTRACT_HELPER.len());
        let path = dir.join(name);
        let up_to_date = fs::metadata(&path).map(|m| m.len() as usize == EXTRACT_HELPER.len()).unwrap_or(false);
        if !up_to_date {
            let tmp = path.with_extension("tmp");
            fs::write(&tmp, EXTRACT_HELPER).ok()?;
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&tmp, fs::Permissions::from_mode(0o755)).ok()?;
            }
            fs::rename(&tmp, &path).ok()?;
        }
        Some(path)
    })
    .clone()
}

fn run_helper(mode: &str, path: &Path) -> Option<std::process::Output> {
    let helper = helper_path()?;
    Command::new(helper).arg(mode).arg(path.as_os_str()).output().ok()
}

fn extract_pdf(path: &Path) -> Result<(usize, String)> {

    match run_helper("pdf", path) {
        Some(out) if out.status.success() => {
            let text_out = String::from_utf8_lossy(&out.stdout).to_string();
            let mut pages = 1;
            let mut content = String::new();

            if let Some(pages_idx) = text_out.find("PAGES:") {
                let rest = &text_out[pages_idx + 6..];
                if let Some(nl) = rest.find('\n') {
                    if let Ok(parsed_pages) = rest[..nl].trim().parse::<usize>() {
                        pages = parsed_pages;
                    }
                }
            }

            if let Some(content_idx) = text_out.find("---CONTENT---") {
                content = text_out[content_idx + 13..].trim().to_string();
            }

            if content.is_empty() {
                content = "(PDF contains 0 extracted text characters. Scanned document or vector paths.)".to_string();
            }

            Ok((pages, content))
        }
        _ => {
            // Fallback: try reading raw strings from PDF or basic summary
            let raw_bytes = fs::read(path)?;
            let text_sample = String::from_utf8_lossy(&raw_bytes);
            let mut extracted = String::new();
            for word in text_sample.split_whitespace() {
                if word.chars().all(|c| c.is_ascii_graphic()) && word.len() > 3 {
                    extracted.push_str(word);
                    extracted.push(' ');
                }
            }
            if extracted.len() > 2000 {
                extracted.truncate(2000);
            }
            let msg = if extracted.is_empty() {
                format!("(PDF document '{}', {} bytes)", path.display(), raw_bytes.len())
            } else {
                format!("(PDF stream extract):\n{extracted}")
            };
            Ok((1, msg))
        }
    }
}

fn extract_image(path: &Path) -> Result<(String, String)> {
    // 1. Get pixel dimensions via sips
    let mut dimensions = String::new();
    if let Ok(sips_out) = Command::new("sips")
        .arg("-g")
        .arg("pixelWidth")
        .arg("-g")
        .arg("pixelHeight")
        .arg(path.as_os_str())
        .output()
    {
        if sips_out.status.success() {
            let sips_str = String::from_utf8_lossy(&sips_out.stdout);
            let mut w = "";
            let mut h = "";
            for line in sips_str.lines() {
                if line.contains("pixelWidth:") {
                    w = line.split(':').nth(1).unwrap_or("").trim();
                } else if line.contains("pixelHeight:") {
                    h = line.split(':').nth(1).unwrap_or("").trim();
                }
            }
            if !w.is_empty() && !h.is_empty() {
                dimensions = format!("{w}x{h}");
            }
        }
    }

    // 2. Apple Vision OCR via the embedded helper

    let recognized_text = match run_helper("ocr", path) {
        Some(out) if out.status.success() => {
            let ocr_out = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if ocr_out.is_empty() {
                "(No text detected in image by Apple Vision OCR)".to_string()
            } else {
                ocr_out
            }
        }
        _ => "(Apple Vision OCR unavailable or failed to process image)".to_string(),
    };

    Ok((dimensions, recognized_text))
}

fn extract_document(path: &Path) -> Result<String> {
    // Try macOS native textutil
    if let Ok(out) = Command::new("/usr/bin/textutil")
        .arg("-convert")
        .arg("txt")
        .arg("-stdout")
        .arg(path.as_os_str())
        .output()
    {
        if out.status.success() {
            let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !text.is_empty() {
                if text.chars().count() > 25000 {
                    let truncated: String = text.chars().take(25000).collect();
                    return Ok(format!(
                        "{truncated}\n\n[... Document truncated at 25,000 chars to fit model context ...]"
                    ));
                }
                return Ok(text);
            }
        }
    }

    // Fallback: direct read
    extract_code_or_text(path)
}

fn extract_code_or_text(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("Failed to read file: {}", path.display()))?;
    let text = String::from_utf8_lossy(&bytes).to_string();
    if text.chars().count() > 25000 {
        let truncated: String = text.chars().take(25000).collect();
        Ok(format!(
            "{truncated}\n\n[... File truncated at 25,000 characters to fit model context ...]"
        ))
    } else {
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_path() {
        let p1 = Path::new("'/tmp/test.rs'");
        assert_eq!(resolve_path(p1).unwrap(), PathBuf::from("/tmp/test.rs"));

        let p2 = Path::new("\"/tmp/my\\ file.txt\"");
        assert_eq!(resolve_path(p2).unwrap(), PathBuf::from("/tmp/my file.txt"));
    }

    #[test]
    fn test_attachment_from_source_code() {
        let temp_file = std::env::temp_dir().join("test_nirvana_code.rs");
        fs::write(&temp_file, "fn main() {\n    println!(\"hello\");\n}\n").unwrap();

        let att = Attachment::from_file(&temp_file).unwrap();
        assert_eq!(att.file_type, AttachmentType::Code);
        assert!(att.extracted_text.contains("hello"));

        let prompt = att.format_prompt("Explain this code");
        assert!(prompt.contains("[ATTACHED FILE: test_nirvana_code.rs"));
        assert!(prompt.contains("Explain this code"));

        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_attachment_from_base64_with_newlines() {
        // "fn main() { println!(\"from base64\"); }" base64 encoded with newlines/spaces
        let raw = "fn main() { println!(\"from base64\"); }";
        let b64 = BASE64_STANDARD.encode(raw);
        let b64_with_newlines = format!("  \n{b64} \r\n");
        let data_url = format!("data:text/plain;base64,{b64_with_newlines}");

        let att = Attachment::from_base64("sample.rs", &data_url).unwrap();
        assert_eq!(att.filename, "sample.rs");
        assert!(att.extracted_text.contains("from base64"));
    }

    #[test]
    fn test_attachment_pdf_extraction() {
        // Any text PDF works: NIRVANA_TEST_PDF=/path/to/file.pdf cargo test
        let Some(pdf_path) = std::env::var_os("NIRVANA_TEST_PDF").map(PathBuf::from) else { return };
        if pdf_path.exists() {
            let att = Attachment::from_file(&pdf_path).unwrap();
            assert_eq!(att.file_type, AttachmentType::Pdf);
            assert!(att.metadata_summary.contains("page"), "{}", att.metadata_summary);
            assert!(!att.extracted_text.is_empty());
        }
    }
}

