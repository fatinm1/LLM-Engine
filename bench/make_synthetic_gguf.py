"""Writes a GGUF file with Llama 3.2 1B shapes and random Q8_0 weights.

LLM Engine dequantizes every weight to FP32 at load time, so its inference
speed depends on tensor shapes, not on the values or the on-disk quant type.
This lets us benchmark the engine without downloading the real model.
"""
import sys

import numpy as np
import gguf

out = sys.argv[1]
n_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 16

# Llama 3.2 1B Instruct hyperparameters
D, FF, H, KH, V, CTX = 2048, 8192, 32, 8, 128256, 131072
Dh = D // H
rng = np.random.default_rng(0)
Q8 = gguf.GGMLQuantizationType.Q8_0

w = gguf.GGUFWriter(out, "llama")
w.add_block_count(n_layers)
w.add_context_length(CTX)
w.add_embedding_length(D)
w.add_feed_forward_length(FF)
w.add_head_count(H)
w.add_head_count_kv(KH)
w.add_rope_freq_base(500000.0)
w.add_vocab_size(V)

# Tokenizer: real special token ids, printable single characters, filler.
tokens = [f"<filler_{i}>" for i in range(V)]
types = [1] * V
for i, c in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'-:;\n"):
    tokens[i] = c
for i, word in enumerate("the of and a to in is was history printing press from its invention modern day write detailed multi paragraph you are helpful assistant".split()):
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


def q8(rows, cols):
    a = (rng.standard_normal((rows, cols), dtype=np.float32) * 0.02)
    return gguf.quants.quantize(a, Q8)


def add(name, rows, cols):
    w.add_tensor(name, q8(rows, cols), raw_dtype=Q8)


add("token_embd.weight", V, D)  # tied output, like Llama 3.2 1B
w.add_tensor("output_norm.weight", np.ones(D, dtype=np.float32))
for L in range(n_layers):
    p = f"blk.{L}."
    w.add_tensor(p + "attn_norm.weight", np.ones(D, dtype=np.float32))
    w.add_tensor(p + "ffn_norm.weight", np.ones(D, dtype=np.float32))
    add(p + "attn_q.weight", H * Dh, D)
    add(p + "attn_k.weight", KH * Dh, D)
    add(p + "attn_v.weight", KH * Dh, D)
    add(p + "attn_output.weight", D, H * Dh)
    add(p + "ffn_gate.weight", FF, D)
    add(p + "ffn_up.weight", FF, D)
    add(p + "ffn_down.weight", D, FF)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out)
