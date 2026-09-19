#!/usr/bin/env bash
# Portable Apple Silicon release build.
#
# .cargo/config.toml sets target-cpu=native for local builds, which can emit
# instructions (e.g. SME on M4) that older chips lack. Distributed binaries
# are built against the M1 baseline instead; ggml still dispatches the best
# CPU kernels at runtime and Metal is unaffected.
#
# Usage: scripts/build-release.sh [--sign "Developer ID Application: ..."]
set -euo pipefail
cd "$(dirname "$0")/.."

SIGN_ID=""
if [[ "${1:-}" == "--sign" ]]; then SIGN_ID="${2:?identity}"; fi

export RUSTFLAGS="-C target-cpu=apple-m1"
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
