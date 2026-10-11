# Synthetic results: Linux VM, 2 vCPU Intel Xeon @ 2.10 GHz (AVX2), 8 GB RAM

Measured on 2026-10-10. No real model download was possible on this machine, so
these runs use random weights with Llama 3.2 1B shapes (16 layers, dim 2048,
FFN 8192, 32 heads, 8 KV heads, vocab 128256). LLM Engine dequantizes all
weights to FP32 at load, so speed depends on shapes, not on weight values.

## 1. Full engine (`llm_bench`) on reduced layer counts

The full 16 layer model was killed by the OOM killer on this 8 GB machine, so
the engine was run with 2 and 8 layers and the per layer cost was fitted.

| layers | decode tok/s | s/token | peak RSS |
| --- | --- | --- | --- |
| 2 | 7.99 | 0.125 | 2927 MB |
| 8 | 4.02 | 0.249 | 4785 MB |

Fit: 0.0206 s per layer + 0.084 s fixed (mostly the 128256 x 2048 output
projection). Predicted for 16 layers: **0.413 s/token (2.4 tok/s)** and
**~7.3 GB peak RSS** (about 310 MB per layer: 243 MB of FP32 weights plus
mapped file pages).

## 2. Matrix vector products only (`matvec_bench`), 3 runs

| variant | tok/s (runs) | median tok/s | weight GB read per token |
| --- | --- | --- | --- |
| fp32_1t (current design) | 2.51, 2.44, 2.32 | 2.44 | 4.94 |
| fp32_mt (rows split over 2 threads) | 4.92, 5.03, 4.44 | 4.92 | 4.94 |
| q8_1t (weights stay Q8_0, int8 dot) | 5.83, 5.69, 4.47 | 5.69 | 1.39 |
| q8_mt (both) | 11.34, 8.01, 10.56 | 10.56 | 1.39 |

The matvec only estimate for the current design (median 0.41 s/token) matches
the full engine fit (0.413 s/token), so matrix vector products account for
nearly all of decode time. Q8_0 vs FP32 relative error on the output projection: 0.0076.

The VM is shared, so run to run variation is noticeable; use the medians.
