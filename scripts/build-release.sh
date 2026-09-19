#!/usr/bin/env bash
# Portable Apple Silicon release build.
#
# .cargo/config.toml sets target-cpu=native for local builds, which can emit
# instructions (e.g. SME on M4) that older chips lack. Distributed binaries
# clear RUSTFLAGS instead: rustc's default CPU for aarch64-apple-darwin is
# already apple-m1, and llama-cpp-sys-2 then builds ggml with GGML_NATIVE=OFF
# for the ARM baseline given below (M1 supports it; Metal is unaffected).
# Note: `-C target-cpu=apple-m1` must NOT be passed — the sys crate forwards
# it to clang as -march=apple-m1, which clang rejects.
#
# Usage: scripts/build-release.sh [--sign "Developer ID Application: ..."]
set -euo pipefail
cd "$(dirname "$0")/.."

SIGN_ID=""
if [[ "${1:-}" == "--sign" ]]; then SIGN_ID="${2:?identity}"; fi

export RUSTFLAGS=""
export GGML_CPU_ARM_ARCH="armv8.4-a+dotprod+fp16"
cargo build --release --locked

BIN=target/release/nirvana-code
VERSION=$(grep -m1 '^version' Cargo.toml | sed -E 's/.*"(.*)"/\1/')
OUT="dist/nirvana-code-${VERSION}-aarch64-apple-darwin"
mkdir -p dist
cp "$BIN" "$OUT"

if [[ -n "$SIGN_ID" ]]; then
  codesign --force --options runtime --timestamp --sign "$SIGN_ID" "$OUT"
  echo "Signed. To notarize:"
  echo "  ditto -c -k --keepParent $OUT $OUT.zip"
  echo "  xcrun notarytool submit $OUT.zip --keychain-profile <profile> --wait"
else
  codesign --force --sign - "$OUT"   # ad-hoc: required for arm64 to run at all
fi

shasum -a 256 "$OUT" | tee "$OUT.sha256"
echo "Built $OUT"
