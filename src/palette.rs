#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum PaletteAction {
    AttachFile,
    DetachFile,
    SelectTemplate(String),
    SelectModel(String),
    ClearHistory,
    CopyLastResponse,
    CopyFullConversation,
    CopyCodeSnippet(usize),
    ToggleSidebar,
    ToggleSpeculative,
    ToggleKvQuantization,
    OpenDocs,
    Quit,
}

#[derive(Debug, Clone)]
pub struct PaletteItem {
    pub title: String,
    pub subtitle: String,
    pub category: &'static str,
    pub action: PaletteAction,
}

pub struct PaletteManager;

impl PaletteManager {
    pub fn build_items(installed_models: &[(std::path::PathBuf, String, u64)]) -> Vec<PaletteItem> {
        // 1. Actions & Controls
        let mut items = vec![PaletteItem {
            title: "Attach File (PDF, Image, Doc, Code)".to_string(),
            subtitle: "Attach local PDF, screenshot, doc, or code file [Ctrl+F or /attach]"
                .to_string(),
            category: "Input",
            action: PaletteAction::AttachFile,
        }];

        items.push(PaletteItem {
            title: "Detach Current Attached File".to_string(),
            subtitle: "Remove attached document/image from upcoming prompt [/detach]".to_string(),
            category: "Input",
            action: PaletteAction::DetachFile,
        });

        items.push(PaletteItem {
            title: "Clear Session & Prefix Cache".to_string(),
            subtitle: "Wipe KV cache context and restart from fresh silicon state".to_string(),
            category: "Controls",
            action: PaletteAction::ClearHistory,
        });

        items.push(PaletteItem {
            title: "Toggle Sidebar (Full Workspace)".to_string(),
            subtitle: "Show or hide the left hardware & templates sidebar [Ctrl+B]".to_string(),
            category: "View",
            action: PaletteAction::ToggleSidebar,
        });

        items.push(PaletteItem {
            title: "Copy Full Assistant Response".to_string(),
            subtitle: "Copy the entire recent generated response to system clipboard [Ctrl+O]"
                .to_string(),
            category: "Controls",
            action: PaletteAction::CopyLastResponse,
        });

        items.push(PaletteItem {
            title: "Copy Full Conversation History".to_string(),
            subtitle: "Copy entire multi-turn conversation and code to system clipboard"
                .to_string(),
            category: "Controls",
            action: PaletteAction::CopyFullConversation,
        });

        items.push(PaletteItem {
            title: "Copy First Code Snippet".to_string(),
            subtitle: "Copy the first code block to system clipboard [Ctrl+Y]".to_string(),
            category: "Controls",
            action: PaletteAction::CopyCodeSnippet(0),
        });

        items.push(PaletteItem {
            title: "Toggle Speculative Decoding".to_string(),
            subtitle: "Enable/disable dual-model speculative draft verification".to_string(),
            category: "Silicon",
            action: PaletteAction::ToggleSpeculative,
        });

        items.push(PaletteItem {
            title: "Toggle KV Cache Quantization (Q8_0 / F16)".to_string(),
            subtitle: "Switch between 50% memory-saving Q8_0 and standard F16 KV cache".to_string(),
            category: "Silicon",
            action: PaletteAction::ToggleKvQuantization,
        });

        // 2. Prompt Engineering & Reasoning Templates
        items.push(PaletteItem {
            title: "Template: Offline Coding Assistant".to_string(),
            subtitle: "Ultra-fast direct coding companion for everyday engineering".to_string(),
            category: "Templates",
            action: PaletteAction::SelectTemplate("offline-assistant".to_string()),
        });

        items.push(PaletteItem {
            title: "Template: Claude 3.7 Sonnet Hybrid Reasoning".to_string(),
            subtitle: "Dual-phase reasoning trace with authoritative architectural synthesis"
                .to_string(),
            category: "Templates",
            action: PaletteAction::SelectTemplate("claude-37-hybrid".to_string()),
        });

        items.push(PaletteItem {
            title: "Template: Antigravity 2.0 / Gemini Flash Thinking".to_string(),
            subtitle: "Multi-step agentic planning with proactive verification".to_string(),
            category: "Templates",
            action: PaletteAction::SelectTemplate("antigravity-20".to_string()),
        });

        items.push(PaletteItem {
            title: "Template: DeepSeek R1 / V3 Reasoning".to_string(),
            subtitle: "Rigorous step-by-step chain-of-thought with self-correction".to_string(),
            category: "Templates",
            action: PaletteAction::SelectTemplate("deepseek-r1".to_string()),
        });

        items.push(PaletteItem {
            title: "Template: OpenAI o1 / o3-mini CoT".to_string(),
            subtitle: "Constraint-dense prompt without conversational filler".to_string(),
            category: "Templates",
            action: PaletteAction::SelectTemplate("openai-o1-o3".to_string()),
        });

        items.push(PaletteItem {
            title: "Template: Structured JSON Schema Extractor".to_string(),
            subtitle: "Deterministic JSON object extraction without preamble".to_string(),
            category: "Templates",
            action: PaletteAction::SelectTemplate("structured-json".to_string()),
        });

        items.push(PaletteItem {
            title: "Template: Code Refactor & Security Audit".to_string(),
            subtitle: "Identify algorithmic bottlenecks and produce clean refactored code"
                .to_string(),
            category: "Templates",
            action: PaletteAction::SelectTemplate("clean-refactor".to_string()),
        });

        // 3. Installed Models
        for (path, name, size) in installed_models {
            let size_mb = size / (1024 * 1024);
            items.push(PaletteItem {
                title: format!("Model: {name}"),
                subtitle: format!("{} MB | {}", size_mb, path.display()),
                category: "Models",
                action: PaletteAction::SelectModel(path.to_string_lossy().to_string()),
            });
        }

        items
    }

    pub fn filter_items<'a>(items: &'a [PaletteItem], query: &str) -> Vec<&'a PaletteItem> {
        if query.trim().is_empty() {
            return items.iter().collect();
        }
        let q = query.to_lowercase();
        items
            .iter()
            .filter(|item| {
                item.title.to_lowercase().contains(&q)
                    || item.subtitle.to_lowercase().contains(&q)
                    || item.category.to_lowercase().contains(&q)
            })
            .collect()
    }
}
