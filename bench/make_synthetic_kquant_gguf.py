"""Writes a GGUF with Llama 3.2 1B's shapes and the Q4_K_M mix of weight types.

Unlike make_synthetic_gguf.py (Q8_0 only), this exercises the Q4_K and Q6_K
kernels and the rope_freqs path, so the quantized engine can be benchmarked
and checked against its FP32 reference mode without downloading the real
model. Weights are random but valid blocks, scaled so the values look like
real weights (roughly +-0.05).

Usage: python3 make_synthetic_kquant_gguf.py out.gguf [n_layers]
"""
import math
import sys

import gguf
import numpy as np

out = sys.argv[1]
n_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 16

D, FF, H, KH, V, CTX = 2048, 8192, 32, 8, 128256, 131072
Dh = D // H
rng = np.random.default_rng(0)
Q4K, Q6K = gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q6_K


def fp16_bytes(values):
    return np.asarray(values, dtype=np.float16).view(np.uint8).reshape(-1, 2)


def q4k(rows, cols):
    nb = rows * cols // 256
    blk = rng.integers(0, 256, size=(nb, 144), dtype=np.uint8)
    # d * scale * q, with scale <= 63 and q <= 15, should reach ~0.05.
    blk[:, 0:2] = fp16_bytes(rng.uniform(3e-5, 6e-5, nb))
    blk[:, 2:4] = fp16_bytes(rng.uniform(3e-5, 6e-5, nb))
    return blk.reshape(rows, cols // 256 * 144)


def q6k(rows, cols):
    nb = rows * cols // 256
    blk = rng.integers(0, 256, size=(nb, 210), dtype=np.uint8)
    # d * |scale| * |q-32| with |scale| <= 127 and |q-32| <= 32.
    blk[:, 208:210] = fp16_bytes(rng.uniform(6e-6, 1.2e-5, nb))
    return blk.reshape(rows, cols // 256 * 210)


w = gguf.GGUFWriter(out, "llama")
w.add_block_count(n_layers)
w.add_context_length(CTX)
w.add_embedding_length(D)
w.add_feed_forward_length(FF)
w.add_head_count(H)
w.add_head_count_kv(KH)
w.add_rope_freq_base(500000.0)
w.add_vocab_size(V)

tokens = [f"<filler_{i}>" for i in range(V)]
types = [1] * V
for i, c in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-:;\n"):
    tokens[i] = c
for i, word in enumerate("the of and a to in is was how language model generates text step by explain "
                         "transformer history printing press you are helpful assistant write".split()):
    tokens[200 + i] = "▁" + word
specials = {128000: "<|begin_of_text|>", 128001: "<|end_of_text|>",
            128006: "<|start_header_id|>", 128007: "<|end_header_id|>",
            128008: "<|eom_id|>", 128009: "<|eot_id|>"}
for i, s in specials.items():
    tokens[i] = s
    types[i] = 3
w.add_tokenizer_model("llama")
w.add_token_list(tokens)
w.add_token_scores([0.0] * V)
w.add_token_types(types)
w.add_bos_token_id(128000)
w.add_eos_token_id(128009)


def add(name, data, qtype):
    w.add_tensor(name, data, raw_dtype=qtype)


# Llama 3 RoPE scaling factors (factor 8, low 1, high 4, original ctx 8192),
# the same values convert_hf_to_gguf.py writes to rope_freqs.weight.
factors = []
for i in range(Dh // 2):
    freq = 500000.0 ** (-2.0 * i / Dh)
    wavelen = 2 * math.pi / freq
    if wavelen < 8192 / 4:
        factors.append(1.0)
    elif wavelen > 8192 / 1:
        factors.append(8.0)
    else:
        smooth = (8192 / wavelen - 1) / (4 - 1)
        factors.append(1.0 / ((1 - smooth) / 8.0 + smooth))
w.add_tensor("rope_freqs.weight", np.array(factors, dtype=np.float32))

add("token_embd.weight", q6k(V, D), Q6K)  # tied output, Q6_K as in Q4_K_M
w.add_tensor("output_norm.weight", np.ones(D, dtype=np.float32))
for L in range(n_layers):
    p = f"blk.{L}."
    w.add_tensor(p + "attn_norm.weight", np.ones(D, dtype=np.float32))
    w.add_tensor(p + "ffn_norm.weight", np.ones(D, dtype=np.float32))
    add(p + "attn_q.weight", q4k(H * Dh, D), Q4K)
    add(p + "attn_k.weight", q4k(KH * Dh, D), Q4K)
    # Q4_K_M keeps attn_v and ffn_down at Q6_K in some layers.
    add(p + "attn_v.weight", q6k(KH * Dh, D) if L % 2 == 0 else q4k(KH * Dh, D), Q6K if L % 2 == 0 else Q4K)
    add(p + "attn_output.weight", q4k(D, H * Dh), Q4K)
    add(p + "ffn_gate.weight", q4k(FF, D), Q4K)
    add(p + "ffn_up.weight", q4k(FF, D), Q4K)
    add(p + "ffn_down.weight", q6k(D, FF) if L % 2 == 0 else q4k(D, FF), Q6K if L % 2 == 0 else Q4K)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out)
