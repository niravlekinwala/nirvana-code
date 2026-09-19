//! Compiles the Swift attachment helper (PDFKit text + Vision OCR) once at
//! build time so the binary can embed it. Needs `swiftc` from the Xcode
//! Command Line Tools — already required to build llama.cpp. If it is
//! missing the helper is left empty and attachments fall back to plain-text
//! extraction at runtime.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let out = out_dir.join("nirvana-extract");
    let src = "helpers/nirvana-extract.swift";
    println!("cargo:rerun-if-changed={src}");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        std::fs::write(&out, []).expect("write empty helper");
        return;
    }

    let status = Command::new("swiftc")
        .args(["-O", "-o"])
        .arg(&out)
        .arg(src)
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!("cargo:warning=swiftc failed ({s}); attachment PDF/OCR helper disabled");
            std::fs::write(&out, []).expect("write empty helper");
        }
        Err(e) => {
            println!("cargo:warning=swiftc not found ({e}); attachment PDF/OCR helper disabled");
            std::fs::write(&out, []).expect("write empty helper");
        }
    }
}
