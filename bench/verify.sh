#!/usr/bin/env bash
# Correctness checks on a real GGUF model.
#
# Usage:
#   bench/verify.sh <model.gguf> [path/to/llama-tokenize]
#
# 1. Kernels: runs llm_compare, which checks the quantized kernels against the
#    FP32 reference path (same tokens, logits compared step by step).
# 2. Tokenizer: compares LLM Engine's token ids with llama.cpp's for a set of
#    strings. Needs llama-tokenize (`brew install llama.cpp`); skipped otherwise.
#
# Exits non-zero if either check fails. Results go to bench/results/<timestamp>/.

set -uo pipefail

MODEL="${1:?usage: bench/verify.sh <model.gguf> [llama-tokenize]}"
LLAMA_TOKENIZE="${2:-$(command -v llama-tokenize || true)}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/bench/results/verify-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
status=0

echo "== kernels: quantized vs FP32 reference"
if "$ROOT/build/llm_compare" "$MODEL" --tokens 64 | tee "$OUT/kernels.txt"; then
  echo
else
  status=1
fi

if [[ -z "$LLAMA_TOKENIZE" ]]; then
  echo "llama-tokenize not found; skipping the tokenizer check."
  exit $status
fi

echo "== tokenizer: LLM Engine vs llama.cpp"
STRINGS=(
  "Hello world"
  "The quick brown fox jumps over the lazy dog."
  "I'm testing contractions: don't, won't, it's."
  "Numbers 12345 and 3.14159, plus 2026-11-15."
  "  leading and trailing spaces  "
  "Tabs	and
newlines"
  "CamelCaseIdentifiers and snake_case_names()"
  "Unicode: café, naïve, 東京, emoji 🚀"
)
pass=0
total=${#STRINGS[@]}
{
  for s in "${STRINGS[@]}"; do
    ours=$("$ROOT/build/llm_compare" "$MODEL" --tokenize "$s" | tr -d ' []')
    theirs=$("$LLAMA_TOKENIZE" -m "$MODEL" -p "$s" --ids --log-disable 2>/dev/null | tail -1 | tr -d ' []')
    if [[ "$ours" == "$theirs" ]]; then
      echo "MATCH     $(printf '%q' "$s")"
      pass=$((pass + 1))
    else
      echo "MISMATCH  $(printf '%q' "$s")"
      echo "  ours:   [$ours]"
      echo "  llama:  [$theirs]"
    fi
  done
  echo
  echo "tokenizer: $pass / $total strings match llama.cpp"
} | tee "$OUT/tokenizer.txt"

if ! grep -q "tokenizer: $total / $total" "$OUT/tokenizer.txt"; then
  status=1
fi
echo "Saved to $OUT"
exit $status
