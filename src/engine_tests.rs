//! Engine tests that need a real GGUF. Skipped unless `NIRVANA_TEST_MODEL`
//! points at a model file (a 0.5B quant is plenty), and `#[ignore]`d so the
//! default `cargo test` stays fast:
//!
//! ```sh
//! NIRVANA_TEST_MODEL=~/.nirvana/models/qwen2.5-0.5b-instruct-q4_k_m.gguf cargo test -- --ignored
//! ```
//!
//! `NIRVANA_TEST_DRAFT` optionally names a separate draft model; by default the
//! target is used as its own draft, which must give ~100% acceptance.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};

use crate::engine::{GenerationConfig, KvQuantMode, ModelEngine, StreamEvent};
use crate::speculative::SpeculativeEngine;

const N_CTX: u32 = 1024;

fn model_path() -> Option<PathBuf> {
    let p = std::env::var_os("NIRVANA_TEST_MODEL")?;
    let p = PathBuf::from(p);
    p.exists().then_some(p)
}

fn draft_path(target: &Path) -> PathBuf {
    std::env::var_os("NIRVANA_TEST_DRAFT")
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .unwrap_or_else(|| target.to_path_buf())
}

struct Run {
    text: String,
    total_tokens: usize,
    prefix_reused: usize,
    kv_type: String,
}

fn drain(mut rx: UnboundedReceiver<StreamEvent>) -> Run {
    let mut run = Run { text: String::new(), total_tokens: 0, prefix_reused: 0, kv_type: String::new() };
    while let Ok(ev) = rx.try_recv() {
        match ev {
            StreamEvent::Token(t) => run.text.push_str(&t),
            StreamEvent::Stats { total_tokens, prefix_tokens_reused, kv_type, .. } => {
                run.total_tokens = total_tokens;
                run.prefix_reused = prefix_tokens_reused;
                run.kv_type = kv_type;
            }
            StreamEvent::Error(e) => panic!("engine error: {e}"),
            StreamEvent::Done => break,
        }
    }
    run
}

fn greedy(max_tokens: usize, ngram: bool) -> GenerationConfig {
    GenerationConfig {
        max_tokens,
        temperature: 0.0,
        use_ngram_speculative: ngram,
        seed: Some(1),
        ..GenerationConfig::default()
    }
}

fn prompt(user: &str) -> String {
    format!("<|im_start|>system\nYou are a terse assistant.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
}

fn run_gguf(engine: &ModelEngine, prompt: &str, config: &GenerationConfig) -> Run {
    let (tx, rx) = unbounded_channel();
    engine
        .stream_generate_with_config(prompt, config, Arc::new(AtomicBool::new(false)), tx)
        .expect("generation failed");
    drain(rx)
}

fn run_spec(engine: &SpeculativeEngine, prompt: &str, config: &GenerationConfig) -> Run {
    let (tx, rx) = unbounded_channel();
    engine
        .stream_generate(prompt, config, Arc::new(AtomicBool::new(false)), tx)
        .expect("generation failed");
    drain(rx)
}

fn load_gguf(path: &Path) -> ModelEngine {
    ModelEngine::load(path, 99, false, KvQuantMode::F16, N_CTX).expect("model load")
}

// A prompt whose answer repeats itself, so prompt-lookup has n-grams to hit.
const REPETITIVE: &str = "List the numbers 1 to 10 as 'Item N: number N' lines, then list them again identically.";

#[test]
#[ignore]
fn regenerate_reuses_prefix_and_is_deterministic() {
    let Some(path) = model_path() else { return };
    let engine = load_gguf(&path);
    let p = prompt("Write one sentence about the sea.");
    let cfg = greedy(24, false);

    let first = run_gguf(&engine, &p, &cfg);
    let second = run_gguf(&engine, &p, &cfg);

    assert!(!first.text.is_empty());
    assert_eq!(first.text, second.text, "greedy regenerate must be deterministic");
    assert_eq!(first.prefix_reused, 0, "cold cache should reuse nothing");
    // Full hit: everything but the last prompt token is reused
    let n_prompt = engine.model.str_to_token(&p, llama_cpp_2::model::AddBos::Always).unwrap().len();
    assert_eq!(second.prefix_reused, n_prompt - 1);
}

#[test]
#[ignore]
fn multi_turn_prefix_reuse_matches_cold_output() {
    let Some(path) = model_path() else { return };
    let engine = load_gguf(&path);
    let cfg = greedy(24, false);

    let turn1 = prompt("Name a colour.");
    let a1 = run_gguf(&engine, &turn1, &cfg);
    let turn2 = format!("{turn1}{}<|im_end|>\n<|im_start|>user\nName another one.<|im_end|>\n<|im_start|>assistant\n", a1.text);
    let warm = run_gguf(&engine, &turn2, &cfg);
    assert!(warm.prefix_reused > 0, "second turn should hit the prefix cache");

    engine.clear_cache();
    let cold = run_gguf(&engine, &turn2, &cfg);
    assert_eq!(cold.prefix_reused, 0);
    assert_eq!(warm.text, cold.text, "warm (prefix-reused) and cold outputs must agree");
}

#[test]
#[ignore]
fn ngram_speculative_matches_plain_greedy() {
    let Some(path) = model_path() else { return };
    let engine = load_gguf(&path);
    let p = prompt(REPETITIVE);

    let plain = run_gguf(&engine, &p, &greedy(96, false));
    let spec = run_gguf(&engine, &p, &greedy(96, true));

    assert_eq!(plain.total_tokens, spec.total_tokens);
    assert_eq!(
        plain.text, spec.text,
        "prompt-lookup decoding changed the output\n--- plain ---\n{}\n--- ngram ---\n{}",
        plain.text, spec.text
    );
}

#[test]
#[ignore]
fn draft_speculative_matches_plain_greedy() {
    let Some(path) = model_path() else { return };
    let draft = draft_path(&path);
    let plain_engine = load_gguf(&path);
    let spec_engine = SpeculativeEngine::load(&path, &draft, 99, false, N_CTX, 4).expect("spec load");
    let p = prompt(REPETITIVE);
    let cfg = greedy(96, false);

    let plain = run_gguf(&plain_engine, &p, &cfg);
    let spec = run_spec(&spec_engine, &p, &cfg);

    assert_eq!(
        plain.text, spec.text,
        "speculative decoding changed the output\n--- plain ---\n{}\n--- spec ---\n{}",
        plain.text, spec.text
    );
    assert_eq!(plain.total_tokens, spec.total_tokens);

    if draft == path {
        // A draft identical to the target must be accepted almost always
        let acc: f64 = spec
            .kv_type
            .split("acc ")
            .nth(1)
            .and_then(|s| s.trim_end_matches("%)").parse().ok())
            .expect("acceptance in stats label");
        assert!(acc >= 90.0, "self-draft acceptance was only {acc}% ({})", spec.kv_type);
    }

    // Second call must reuse both prefix caches and give the same answer
    let again = run_spec(&spec_engine, &p, &cfg);
    assert!(again.prefix_reused > 0);
    assert_eq!(again.text, spec.text);
}
