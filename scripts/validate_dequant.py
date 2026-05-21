#!/usr/bin/env python3
"""Reference dequantization via gguf library for comparison with C++ dequantizers."""

import sys

import numpy as np

try:
    from gguf import GGUFReader
    from gguf.quants import dequantize
except ImportError:
    print("pip install gguf")
    sys.exit(1)

path = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
reader = GGUFReader(path)

TARGETS = (
    "token_embd.weight",
    "blk.0.attn_q.weight",
    "blk.0.attn_norm.weight",
)

for tensor in reader.tensors:
    if tensor.name not in TARGETS:
        continue

    if tensor.tensor_type.value in (0, 1):  # F32, F16
        data = np.array(tensor.data, dtype=np.float32)
    else:
        data = dequantize(np.array(tensor.data), tensor.tensor_type)

    print(f"{tensor.name}: shape={data.shape} dtype={data.dtype} "
          f"tensor_type={tensor.tensor_type.name}({tensor.tensor_type.value})")
    flat = data.flatten()
    print(f"  min={flat.min():.6f} max={flat.max():.6f}")
    print(f"  first 8: {flat[:8]}")
    print()
