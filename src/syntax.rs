use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

#[derive(Debug, Clone)]
pub enum ContentBlock {
    Text(String),
    Code {
        lang: String,
        code: String,
        index: usize,
    },
}

pub struct SyntaxHighlighter;

impl SyntaxHighlighter {
    pub fn parse_markdown(content: &str) -> Vec<ContentBlock> {
        let mut blocks = Vec::new();
        let mut current_text = String::new();
        let mut current_code = String::new();
        let mut current_lang = String::new();
        let mut in_code = false;
        let mut code_idx = 0;

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("```") {
                if in_code {
                    // End code block
                    blocks.push(ContentBlock::Code {
                        lang: current_lang.clone(),
                        code: current_code.clone(),
                        index: code_idx,
                    });
                    code_idx += 1;
                    current_code.clear();
                    current_lang.clear();
                    in_code = false;
                } else {
                    // Start code block
                    if !current_text.is_empty() {
                        blocks.push(ContentBlock::Text(current_text.clone()));
                        current_text.clear();
                    }
                    current_lang = trimmed.trim_start_matches('`').trim().to_lowercase();
                    if current_lang.is_empty() {
                        current_lang = "text".to_string();
                    }
                    in_code = true;
                }
            } else if in_code {
                current_code.push_str(line);
                current_code.push('\n');
            } else {
                current_text.push_str(line);
                current_text.push('\n');
            }
        }

        if in_code {
            blocks.push(ContentBlock::Code {
                lang: current_lang,
                code: current_code,
                index: code_idx,
            });
        } else if !current_text.is_empty() {
            blocks.push(ContentBlock::Text(current_text));
        }

        blocks
    }

    pub fn highlight_code_line<'a>(line: &'a str, lang: &str) -> Vec<Span<'a>> {
        let comment_prefix = match lang {
            "python" | "bash" | "sh" | "zsh" | "yaml" => "#",
            "sql" => "--",
            _ => "//",
        };

        if let Some(pos) = line.find(comment_prefix) {
            let before = &line[..pos];
            let comment = &line[pos..];
            let mut spans = Self::highlight_tokens(before, lang);
            spans.push(Span::styled(
                comment,
                Style::default().fg(Color::Rgb(100, 115, 135)),
            ));
            return spans;
        }

        Self::highlight_tokens(line, lang)
    }

    fn highlight_tokens<'a>(text: &'a str, lang: &str) -> Vec<Span<'a>> {
        let mut spans = Vec::new();
        let mut current_idx = 0;
        let bytes = text.as_bytes();
        let len = bytes.len();

        let keywords = Self::get_keywords(lang);
        let types = Self::get_types(lang);

        while current_idx < len {
            // String literals
            if bytes[current_idx] == b'"' || bytes[current_idx] == b'\'' {
                let quote = bytes[current_idx];
                let start = current_idx;
                current_idx += 1;
                while current_idx < len && bytes[current_idx] != quote {
                    if bytes[current_idx] == b'\\' && current_idx + 1 < len {
                        current_idx += 2;
                    } else {
                        current_idx += 1;
                    }
                }
                if current_idx < len {
                    current_idx += 1;
                }
                spans.push(Span::styled(
                    &text[start..current_idx],
                    Style::default().fg(Color::Rgb(0, 255, 180)),
                ));
                continue;
            }

            // Word token (identifier / keyword / type)
            if bytes[current_idx].is_ascii_alphabetic() || bytes[current_idx] == b'_' {
                let start = current_idx;
                while current_idx < len
                    && (bytes[current_idx].is_ascii_alphanumeric() || bytes[current_idx] == b'_')
                {
                    current_idx += 1;
                }
                let word = &text[start..current_idx];

                if keywords.contains(&word) {
                    spans.push(Span::styled(
                        word,
                        Style::default()
                            .fg(Color::Rgb(255, 0, 128))
                            .add_modifier(Modifier::BOLD),
                    ));
                } else if types.contains(&word) {
                    spans.push(Span::styled(
                        word,
                        Style::default().fg(Color::Rgb(0, 230, 255)),
                    ));
                } else {
                    spans.push(Span::styled(
                        word,
                        Style::default().fg(Color::Rgb(225, 235, 245)),
                    ));
                }
                continue;
            }

            // Numbers
            if bytes[current_idx].is_ascii_digit() {
                let start = current_idx;
                while current_idx < len
                    && (bytes[current_idx].is_ascii_digit()
                        || bytes[current_idx] == b'.'
                        || bytes[current_idx] == b'_')
                {
                    current_idx += 1;
                }
                spans.push(Span::styled(
                    &text[start..current_idx],
                    Style::default().fg(Color::Rgb(255, 180, 50)),
                ));
                continue;
            }

            // Punctuation / whitespace
            let start = current_idx;
            while current_idx < len
                && !bytes[current_idx].is_ascii_alphanumeric()
                && bytes[current_idx] != b'_'
                && bytes[current_idx] != b'"'
                && bytes[current_idx] != b'\''
            {
                current_idx += 1;
            }
            spans.push(Span::styled(
                &text[start..current_idx],
                Style::default().fg(Color::Rgb(140, 160, 180)),
            ));
        }

        spans
    }

    fn get_keywords(lang: &str) -> &'static [&'static str] {
        match lang {
            "rust" | "rs" => &[
                "fn", "let", "mut", "pub", "struct", "enum", "impl", "trait", "match", "if",
                "else", "for", "while", "loop", "return", "async", "await", "use", "mod", "const",
                "static", "type", "where", "move", "unsafe", "ref", "self", "Self",
            ],
            "python" | "py" => &[
                "def", "class", "return", "if", "elif", "else", "for", "while", "import", "from",
                "as", "try", "except", "finally", "with", "lambda", "yield", "async", "await",
                "pass", "break", "continue", "in", "is", "not", "and", "or",
            ],
            "javascript" | "js" | "typescript" | "ts" => &[
                "function",
                "const",
                "let",
                "var",
                "return",
                "if",
                "else",
                "for",
                "while",
                "import",
                "export",
                "from",
                "default",
                "class",
                "extends",
                "async",
                "await",
                "try",
                "catch",
                "new",
                "this",
                "typeof",
                "interface",
                "type",
            ],
            "c" | "cpp" | "cxx" => &[
                "int",
                "char",
                "void",
                "return",
                "if",
                "else",
                "for",
                "while",
                "class",
                "struct",
                "namespace",
                "using",
                "template",
                "typename",
                "public",
                "private",
                "protected",
                "const",
                "auto",
                "virtual",
                "override",
            ],
            "go" => &[
                "func",
                "package",
                "import",
                "return",
                "if",
                "else",
                "for",
                "range",
                "var",
                "type",
                "struct",
                "interface",
                "go",
                "chan",
                "select",
                "case",
                "default",
            ],
            "sql" => &[
                "SELECT", "FROM", "WHERE", "INSERT", "INTO", "UPDATE", "DELETE", "JOIN", "LEFT",
                "RIGHT", "INNER", "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "CREATE", "TABLE",
                "DROP", "ALTER", "select", "from", "where", "insert", "update", "delete",
            ],
            _ => &["if", "else", "for", "while", "return", "true", "false"],
        }
    }

    fn get_types(lang: &str) -> &'static [&'static str] {
        match lang {
            "rust" | "rs" => &[
                "String", "str", "u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64",
                "isize", "f32", "f64", "bool", "Vec", "Option", "Result", "Some", "None", "Ok",
                "Err", "Arc", "Box", "Rc", "Path", "PathBuf",
            ],
            "python" | "py" => &[
                "int", "str", "float", "bool", "list", "dict", "set", "tuple", "Any", "Optional",
                "List", "Dict", "Union", "None", "True", "False",
            ],
            "typescript" | "ts" => &[
                "string",
                "number",
                "boolean",
                "any",
                "void",
                "null",
                "undefined",
                "Promise",
                "Array",
                "Record",
            ],
            _ => &["int", "float", "double", "bool", "string", "void"],
        }
    }

    pub fn render_markdown_lines(content: &str) -> Vec<Line<'static>> {
        let blocks = Self::parse_markdown(content);
        let mut lines = Vec::new();

        for block in blocks {
            match block {
                ContentBlock::Text(text) => {
                    for raw_line in text.lines() {
                        let trimmed = raw_line.trim();
                        if trimmed.starts_with("# ") {
                            lines.push(Line::from(vec![Span::styled(
                                trimmed.to_string(),
                                Style::default()
                                    .fg(Color::Rgb(0, 240, 255))
                                    .add_modifier(Modifier::BOLD),
                            )]));
                        } else if trimmed.starts_with("## ") {
                            lines.push(Line::from(vec![Span::styled(
                                trimmed.to_string(),
                                Style::default()
                                    .fg(Color::Rgb(255, 0, 128))
                                    .add_modifier(Modifier::BOLD),
                            )]));
                        } else if trimmed.starts_with("### ") {
                            lines.push(Line::from(vec![Span::styled(
                                trimmed.to_string(),
                                Style::default()
                                    .fg(Color::Rgb(255, 180, 50))
                                    .add_modifier(Modifier::BOLD),
                            )]));
                        } else {
                            lines.push(Line::from(vec![Span::styled(
                                raw_line.to_string(),
                                Style::default().fg(Color::Rgb(220, 230, 245)),
                            )]));
                        }
                    }
                }
                ContentBlock::Code { lang, code, index } => {
                    // Header frame for code block
                    let header = format!(
                        "  ⚡ [Snippet #{}] Lang: {}  (Ctrl+Y to copy)",
                        index + 1,
                        lang.to_uppercase()
                    );
                    lines.push(Line::from(vec![Span::styled(
                        header,
                        Style::default()
                            .fg(Color::Rgb(0, 230, 255))
                            .bg(Color::Rgb(16, 26, 38))
                            .add_modifier(Modifier::BOLD),
                    )]));

                    // Code lines with syntax highlighting
                    for (i, code_line) in code.lines().enumerate() {
                        let line_num = format!("{:3} │ ", i + 1);
                        let mut line_spans = vec![Span::styled(
                            line_num,
                            Style::default().fg(Color::Rgb(80, 95, 115)),
                        )];
                        let highlighted = Self::highlight_code_line(code_line, &lang);
                        for span in highlighted {
                            line_spans.push(Span::styled(span.content.to_string(), span.style));
                        }
                        lines.push(Line::from(line_spans));
                    }

                    // Bottom separator
                    lines.push(Line::from(vec![Span::styled(
                        "  └─────────────────────────────────────────────────────",
                        Style::default().fg(Color::Rgb(40, 55, 75)),
                    )]));
                    lines.push(Line::from(""));
                }
            }
        }

        lines
    }

    pub fn extract_code_blocks(content: &str) -> Vec<String> {
        let blocks = Self::parse_markdown(content);
        blocks
            .into_iter()
            .filter_map(|b| match b {
                ContentBlock::Code { code, .. } => Some(code),
                _ => None,
            })
            .collect()
    }
}
