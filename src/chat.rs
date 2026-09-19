//! Backend-neutral chat messages and prompt rendering.
//!
//! GGUF models carry their own Jinja chat template in the metadata; llama.cpp
//! recognises the common ones (ChatML, Llama 3, Gemma, Mistral, DeepSeek, …)
//! and renders them natively. Anything it does not recognise falls back to
//! ChatML, which is what this project hard-coded for every model before.

use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: content.into() }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

/// ChatML rendering, used when the model has no template llama.cpp understands.
pub fn render_chatml(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str("<|im_start|>");
        out.push_str(&m.role);
        out.push('\n');
        out.push_str(&m.content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

/// Which renderer a model will use; resolved once at load.
pub enum ChatFormat {
    /// The model's own Jinja template, rendered faithfully (what llama-server does).
    Jinja {
        env: Box<minijinja::Environment<'static>>,
        bos_token: String,
        eos_token: String,
    },
    /// The model's template as recognised by llama.cpp's built-in matcher.
    Native(LlamaChatTemplate),
    ChatMl,
}

const TEMPLATE_NAME: &str = "chat";

fn special_token_text(model: &LlamaModel, tok: llama_cpp_2::token::LlamaToken) -> String {
    model
        .token_to_piece_bytes(tok, 64, true, None)
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default()
}

fn jinja_env(source: &str) -> Result<minijinja::Environment<'static>, minijinja::Error> {
    let mut env = minijinja::Environment::new();
    // HF templates lean on Python string methods (.strip(), .startswith()…)
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    minijinja_contrib::add_to_environment(&mut env);
    env.add_function("raise_exception", |msg: String| -> Result<(), minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
    });
    env.add_template_owned(TEMPLATE_NAME, source.to_string())?;
    Ok(env)
}

fn jinja_render(
    env: &minijinja::Environment<'static>,
    messages: &[ChatMessage],
    bos_token: &str,
    eos_token: &str,
) -> Result<String, minijinja::Error> {
    let msgs: Vec<minijinja::value::Value> = messages
        .iter()
        .map(|m| minijinja::context! { role => m.role, content => m.content })
        .collect();
    env.get_template(TEMPLATE_NAME)?.render(minijinja::context! {
        messages => msgs,
        add_generation_prompt => true,
        bos_token => bos_token,
        eos_token => eos_token,
    })
}

impl ChatFormat {
    pub fn detect(model: &LlamaModel) -> Self {
        let Ok(tmpl) = model.chat_template(None) else {
            return ChatFormat::ChatMl;
        };
        let probe = [ChatMessage::system("s"), ChatMessage::user("hi")];

        // 1. Full Jinja: handles every template the model ships with.
        if let Ok(src) = tmpl.as_c_str().to_str() {
            if let Ok(env) = jinja_env(src) {
                let bos_token = special_token_text(model, model.token_bos());
                let eos_token = special_token_text(model, model.token_eos());
                if jinja_render(&env, &probe, &bos_token, &eos_token).is_ok() {
                    return ChatFormat::Jinja { env: Box::new(env), bos_token, eos_token };
                }
            }
        }

        // 2. llama.cpp's built-in matcher for the common families.
        let native_probe: Vec<LlamaChatMessage> = probe
            .iter()
            .filter_map(|m| LlamaChatMessage::new(m.role.clone(), m.content.clone()).ok())
            .collect();
        if model.apply_chat_template(&tmpl, &native_probe, true).is_ok() {
            return ChatFormat::Native(tmpl);
        }

        ChatFormat::ChatMl
    }

    pub fn label(&self) -> &'static str {
        match self {
            ChatFormat::Jinja { .. } => "model template (jinja)",
            ChatFormat::Native(_) => "model template (llama.cpp builtin)",
            ChatFormat::ChatMl => "ChatML (fallback)",
        }
    }

    /// Render `messages` followed by an open assistant turn.
    pub fn render(&self, model: &LlamaModel, messages: &[ChatMessage], strip_bos: Option<&str>) -> String {
        let rendered = match self {
            ChatFormat::Jinja { env, bos_token, eos_token } => {
                jinja_render(env, messages, bos_token, eos_token).ok()
            }
            ChatFormat::Native(tmpl) => {
                let chat: Vec<LlamaChatMessage> = messages
                    .iter()
                    .filter_map(|m| LlamaChatMessage::new(m.role.clone(), m.content.clone()).ok())
                    .collect();
                model.apply_chat_template(tmpl, &chat, true).ok()
            }
            ChatFormat::ChatMl => None,
        };
        match rendered {
            Some(mut s) => {
                // Templates like Llama 3 spell out the BOS token; the tokenizer
                // adds it too, so drop the textual copy.
                if let Some(bos) = strip_bos {
                    if !bos.is_empty() && s.starts_with(bos) {
                        s = s[bos.len()..].to_string();
                    }
                }
                s
            }
            None => render_chatml(messages),
        }
    }
}

/// A model's resolved chat format plus the BOS text to strip, with fitting.
pub struct ChatRenderer {
    format: ChatFormat,
    /// Textual BOS token when the vocab auto-inserts BOS, so a template that
    /// spells it out does not produce a double BOS.
    bos_text: Option<String>,
}

impl ChatRenderer {
    pub fn detect(model: &LlamaModel) -> Self {
        // add_special tokenization of "" yields exactly the BOS when the vocab
        // wants one inserted; otherwise it yields nothing.
        let bos_text = model
            .str_to_token("", AddBos::Always)
            .ok()
            .and_then(|t| t.first().copied())
            .and_then(|bos| model.token_to_piece_bytes(bos, 64, true, None).ok())
            .map(|b| String::from_utf8_lossy(&b).into_owned())
            .filter(|s| !s.is_empty());
        Self { format: ChatFormat::detect(model), bos_text }
    }

    pub fn label(&self) -> &'static str {
        self.format.label()
    }

    pub fn render(&self, model: &LlamaModel, messages: &[ChatMessage]) -> String {
        self.format.render(model, messages, self.bos_text.as_deref())
    }

    /// Render, dropping the oldest turns until the prompt leaves room for
    /// `max_tokens` of output inside `n_ctx`. Turns are dropped whole so the
    /// template is never cut mid-message.
    pub fn render_fitting(
        &self,
        model: &LlamaModel,
        messages: &[ChatMessage],
        n_ctx: usize,
        max_tokens: usize,
    ) -> String {
        let limit = n_ctx.saturating_sub(max_tokens.min(n_ctx / 2).max(128));
        let mut msgs = messages.to_vec();
        loop {
            let prompt = self.render(model, &msgs);
            let n = model
                .str_to_token(&prompt, AddBos::Always)
                .map(|t| t.len())
                .unwrap_or(0);
            if n <= limit || !drop_oldest_turn(&mut msgs) {
                return prompt;
            }
        }
    }
}

/// Drop the oldest non-system turn. Returns `false` if nothing can be dropped.
/// Used to make a conversation fit the context without cutting mid-turn.
pub fn drop_oldest_turn(messages: &mut Vec<ChatMessage>) -> bool {
    // Never drop the final message (the current user request).
    let last = messages.len().saturating_sub(1);
    if let Some(idx) = messages[..last].iter().position(|m| m.role != "system") {
        messages.remove(idx);
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_render_matches_previous_hardcoded_format() {
        let msgs = [ChatMessage::system("S"), ChatMessage::user("U")];
        assert_eq!(
            render_chatml(&msgs),
            "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn drop_oldest_keeps_system_and_last() {
        let mut msgs = vec![
            ChatMessage::system("S"),
            ChatMessage::user("u1"),
            ChatMessage::assistant("a1"),
            ChatMessage::user("u2"),
        ];
        assert!(drop_oldest_turn(&mut msgs));
        assert_eq!(msgs, vec![ChatMessage::system("S"), ChatMessage::assistant("a1"), ChatMessage::user("u2")]);
        assert!(drop_oldest_turn(&mut msgs));
        assert_eq!(msgs, vec![ChatMessage::system("S"), ChatMessage::user("u2")]);
        assert!(!drop_oldest_turn(&mut msgs));
    }
}
