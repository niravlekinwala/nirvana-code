//! `~/.config/nirvana-code/config.toml` — persistent defaults for the CLI.
//!
//! Precedence: explicit command-line flag > config file > built-in default.
//! Every key is optional and named after the long flag with dashes replaced
//! by underscores:
//!
//! ```toml
//! model = "qwen-coder-3b"
//! draft_model = "qwen-0.5b"
//! ctx_size = 8192
//! kv_type = "q8_0"
//! temperature = 0.4
//! persist_kv = true
//! workspace = "~/code"
//! api_key = "nv-..."
//! cors_origin = ["http://localhost:5173"]
//! ```

use clap::parser::ValueSource;
use clap::{ArgMatches, CommandFactory};
use serde::Deserialize;
use std::path::PathBuf;

use crate::cli::Cli;

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileConfig {
    pub model: Option<String>,
    pub draft_model: Option<String>,
    pub speculative: Option<bool>,
    pub n_draft: Option<usize>,
    pub kv_type: Option<String>,
    pub no_mlock: Option<bool>,
    pub gpu_layers: Option<u32>,
    pub ctx_size: Option<u32>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub min_p: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub repeat_penalty: Option<f32>,
    pub dry_multiplier: Option<f32>,
    pub ubatch: Option<u32>,
    pub persist_kv: Option<bool>,
    pub seed: Option<u32>,
    pub ngram_speculative: Option<bool>,
    pub verbose: Option<bool>,
    pub workspace: Option<String>,
    pub api_key: Option<String>,
    pub cors_origin: Option<Vec<String>>,
    pub allow_host: Option<Vec<String>>,
}

pub fn config_path() -> Option<PathBuf> {
    std::env::var_os("NIRVANA_CONFIG")
        .map(PathBuf::from)
        .or_else(|| dirs::config_dir().map(|d| d.join("nirvana-code").join("config.toml")))
}

fn expand_tilde(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(p)
}

impl FileConfig {
    pub fn load() -> Option<(PathBuf, Self)> {
        let path = config_path()?;
        let text = std::fs::read_to_string(&path).ok()?;
        match toml::from_str::<FileConfig>(&text) {
            Ok(cfg) => Some((path, cfg)),
            Err(e) => {
                eprintln!("⚠️  Ignoring {}: {e}", path.display());
                None
            }
        }
    }

    /// Fill in every field the user did not set on the command line.
    pub fn apply(self, cli: &mut Cli, matches: &ArgMatches) {
        let from_cli = |id: &str| matches.value_source(id) == Some(ValueSource::CommandLine);

        macro_rules! take {
            ($field:ident, $id:literal) => {
                if !from_cli($id) {
                    if let Some(v) = self.$field {
                        cli.$field = v;
                    }
                }
            };
        }
        macro_rules! take_opt {
            ($field:ident, $id:literal, $map:expr) => {
                if !from_cli($id) && cli.$field.is_none() {
                    if let Some(v) = self.$field {
                        cli.$field = Some($map(v));
                    }
                }
            };
        }

        take_opt!(model, "model", |v: String| expand_tilde(&v));
        take_opt!(draft_model, "draft_model", |v: String| expand_tilde(&v));
        take!(speculative, "speculative");
        take!(n_draft, "n_draft");
        take!(kv_type, "kv_type");
        take!(no_mlock, "no_mlock");
        take!(gpu_layers, "gpu_layers");
        take!(ctx_size, "ctx_size");
        take!(max_tokens, "max_tokens");
        take!(temperature, "temperature");
        take!(min_p, "min_p");
        take!(top_p, "top_p");
        take!(top_k, "top_k");
        take!(repeat_penalty, "repeat_penalty");
        take!(dry_multiplier, "dry_multiplier");
        take_opt!(ubatch, "ubatch", |v| v);
        take!(persist_kv, "persist_kv");
        take_opt!(seed, "seed", |v| v);
        take!(ngram_speculative, "ngram_speculative");
        take!(verbose, "verbose");
        take_opt!(workspace, "workspace", |v: String| expand_tilde(&v));
        take_opt!(api_key, "api_key", |v| v);
        if !from_cli("cors_origin") && cli.cors_origin.is_empty() {
            if let Some(v) = self.cors_origin {
                cli.cors_origin = v;
            }
        }
        if !from_cli("allow_host") && cli.allow_host.is_empty() {
            if let Some(v) = self.allow_host {
                cli.allow_host = v;
            }
        }
    }
}

/// Parse the command line and layer the config file underneath it.
pub fn parse_cli() -> Cli {
    let matches = Cli::command().get_matches();
    let mut cli =
        <Cli as clap::FromArgMatches>::from_arg_matches(&matches).unwrap_or_else(|e| e.exit());
    if let Some((_, cfg)) = FileConfig::load() {
        cfg.apply(&mut cli, &matches);
    }
    cli
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_beats_config_beats_default() {
        let cfg: FileConfig = toml::from_str(
            r#"
            ctx_size = 8192
            temperature = 0.2
            model = "~/m.gguf"
            cors_origin = ["http://a"]
            "#,
        )
        .unwrap();
        let matches = Cli::command().get_matches_from(["nirvana-code", "--temperature", "0.9"]);
        let mut cli = <Cli as clap::FromArgMatches>::from_arg_matches(&matches).unwrap();
        cfg.apply(&mut cli, &matches);
        assert_eq!(cli.temperature, 0.9, "explicit flag wins");
        assert_eq!(cli.ctx_size, 8192, "config fills unset flag");
        assert_eq!(cli.top_k, 40, "built-in default survives");
        assert!(cli.model.unwrap().ends_with("m.gguf"));
        assert_eq!(cli.cors_origin, vec!["http://a".to_string()]);
    }

    #[test]
    fn unknown_keys_are_rejected() {
        assert!(toml::from_str::<FileConfig>("bogus = 1").is_err());
    }
}
