#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$( dirname "$DIR" )"

echo "⚡ [NIRVANA CODE] Reproducible Apple Silicon Benchmark Suite"
echo "   Directory: $PROJECT_ROOT"
echo ""

# 1. Build Nirvana Code in release mode if needed
if [ ! -f "$PROJECT_ROOT/target/release/nirvana-code" ]; then
    echo "📦 Building Nirvana Code (v2) in release mode with native optimizations..."
    (cd "$PROJECT_ROOT" && RUSTFLAGS="-C target-cpu=native" cargo build --release)
fi

# 2. Check Ollama is installed and running
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed. Install with: brew install ollama"
    exit 1
fi

if ! pgrep -x "ollama" > /dev/null; then
    echo "🚀 Starting Ollama background server daemon..."
    ollama serve &
    sleep 3
fi

# 3. Ensure benchmark model is registered in Ollama
MODEL_FILE="$HOME/.promptcraft/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Model file not found at: $MODEL_FILE"
    echo "   Download via: $PROJECT_ROOT/target/release/nirvana-code download qwen-1.5b"
    exit 1
fi

if ! ollama list | grep -q "qwen1.5b-bench"; then
    echo "📦 Registering GGUF model in Ollama as 'qwen1.5b-bench'..."
    TMP_MODELFILE="/tmp/Modelfile.qwen1.5b"
    echo "FROM $MODEL_FILE" > "$TMP_MODELFILE"
    ollama create qwen1.5b-bench -f "$TMP_MODELFILE"
    rm -f "$TMP_MODELFILE"
fi

# 4. Run automated python benchmark suite
python3 "$DIR/run_benchmarks.py"
