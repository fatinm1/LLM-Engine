#!/usr/bin/env bash
# Benchmarks LLM Engine and llama.cpp on the same GGUF file and machine.
#
# Usage:
#   bench/compare.sh <model.gguf> [path/to/llama-bench]
#
# Requires LLM Engine built in Release (build/llm_bench). For llama.cpp, either
# pass the llama-bench path or have it on PATH (for example `brew install llama.cpp`).
# Results are written to bench/results/<timestamp>/.

set -euo pipefail

MODEL="${1:?usage: bench/compare.sh <model.gguf> [llama-bench]}"
LLAMA_BENCH="${2:-$(command -v llama-bench || true)}"
GEN=128
REPS=3

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/bench/results/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"

echo "== machine"
{
  uname -a
  if [[ "$(uname)" == "Darwin" ]]; then
    sysctl -n machdep.cpu.brand_string
    sysctl -n hw.ncpu hw.memsize
  else
    lscpu | grep -E "Model name|^CPU\(s\)"
    free -g | head -2
  fi
} | tee "$OUT/machine.txt"

echo
echo "== llm-engine"
"$ROOT/build/llm_bench" "$MODEL" --gen "$GEN" --reps "$REPS" | tee "$OUT/llm_engine.txt"
"$ROOT/build/llm_bench" "$MODEL" --gen "$GEN" --reps "$REPS" --json > "$OUT/llm_engine.json"

PROMPT_TOKENS=$(sed -n 's/.*"prompt_tokens":\([0-9]*\).*/\1/p' "$OUT/llm_engine.json")

if [[ -z "$LLAMA_BENCH" ]]; then
  echo
  echo "llama-bench not found; skipping llama.cpp. Pass its path as the second argument."
  exit 0
fi

echo
echo "== llama.cpp, default settings (uses the GPU via Metal on Apple Silicon)"
"$LLAMA_BENCH" -m "$MODEL" -p "$PROMPT_TOKENS" -n "$GEN" -r "$REPS" | tee "$OUT/llama_cpp.txt"

echo
echo "== llama.cpp, CPU only, default thread count"
"$LLAMA_BENCH" -m "$MODEL" -p "$PROMPT_TOKENS" -n "$GEN" -r "$REPS" -ngl 0 | tee "$OUT/llama_cpp_cpu.txt"

echo
echo "== llama.cpp, 1 thread, CPU only (closest to LLM Engine's single threaded matvec)"
"$LLAMA_BENCH" -m "$MODEL" -p "$PROMPT_TOKENS" -n "$GEN" -r "$REPS" -t 1 -ngl 0 | tee "$OUT/llama_cpp_1t.txt"

echo
echo "Saved to $OUT"
