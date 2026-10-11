# Why my from scratch LLM engine is slower than llama.cpp, and what I'd fix first

*Draft. Numbers marked [TBD] come from running `bench/compare.sh` on my Mac.*

I built [LLM Engine](https://github.com/fatinm1/LLM-Engine), a C++ inference engine that runs Llama 3.2 on a CPU with no ML libraries: my own GGUF parser, tokenizer, transformer forward pass, KV cache and sampler. It works, but it's much slower than llama.cpp. I wanted to know exactly why, so I measured it.

## Setup

- **Model:** Llama 3.2 1B Instruct, Q4_K_M GGUF (about 808 MB on disk)
- **Mac:** [TBD: chip, cores, RAM]
- **Tools:** `llm_bench`, a small benchmark I added that times prompt processing and generation separately and records peak memory, and llama.cpp's `llama-bench` on the same file
- **Settings:** greedy decoding, 128 generated tokens, median of 3 runs

One thing I found right away: the tokens per second number my app showed included the time spent reading the prompt. With a short reply, that made generation look slower than it is. `llm_bench` reports the two separately.

## Results on my Mac

| engine | prompt tok/s | generation tok/s | peak memory |
| --- | --- | --- | --- |
| LLM Engine | [TBD] | [TBD] | [TBD] |
| llama.cpp, CPU, 1 thread | [TBD] | [TBD] | [TBD] |
| llama.cpp, CPU, all cores | [TBD] | [TBD] | [TBD] |
| llama.cpp, default (Metal GPU) | [TBD] | [TBD] | [TBD] |

## Where the time goes

Generating one token is mostly matrix vector products: each of the 16 layers multiplies the hidden state by seven weight matrices, and then a final 128,256 by 2,048 matrix turns it into scores for every token in the vocabulary.

To see how much of the time that accounts for, I ran two experiments on a Linux machine (2 vCPU Xeon with AVX2) using random weights with exactly Llama 3.2 1B's shapes. My engine converts every weight to FP32 when it loads, so its speed depends on the shapes, not the values.

1. The full engine took **0.413 seconds per token**, measured at 2 and 8 layers and fitted for 16, because 16 layers didn't fit in memory on that machine (more on that below).
2. Just the matrix vector products, done the same way the engine does them, took **0.41 seconds per token**.

So nearly all of generation time is those products. Attention, normalization, RoPE and sampling barely register at this context length.

Three design choices make those products slow:

**1. Every weight is converted to FP32 at load.** The file stores weights in about 5 bits each. My engine expands them to 32 bit floats, so each token reads about **4.9 GB of weights from memory instead of about 0.8 GB**, roughly six times more. Generating a token on a CPU is mostly limited by how fast weights can come out of memory, so this hurts the most.

**2. The heavy math runs on one thread.** I added a thread pool, but it only splits up the attention heads, which are tiny at this sequence length. The matrix vector products, where nearly all the time goes, run on a single core.

**3. No SIMD on Apple Silicon.** On x86 the dot products use AVX2. On my Mac they fall back to plain C++ loops and rely on the compiler to vectorize them. llama.cpp has hand written NEON kernels, and Metal when the GPU is used.

## The memory surprise

Expanding to FP32 also explains memory use. For the 1B model, the FP32 weights alone are about **6 GB**:

- 3.9 GB for the 16 layers
- 1.05 GB for the token embeddings
- another 1.05 GB because the output layer shares its weights with the embeddings, but my code copies that buffer instead of pointing to it

On an 8 GB Linux machine, the full model was killed for running out of memory. Fitting the 2 and 8 layer runs predicts about 7.3 GB peak for all 16 layers. My README said "~1.2 GB," which was wrong. llama.cpp keeps the weights quantized and memory maps the file, so it stays close to the file size.

## What fixing it could look like

On the same Linux machine I benchmarked the matrix vector products four ways (median of 3 runs, matrix math only):

| variant | tok/s | weight data read per token |
| --- | --- | --- |
| current: FP32, 1 thread | 2.4 | 4.94 GB |
| FP32, rows split across 2 threads | 4.9 | 4.94 GB |
| keep weights in Q8_0, int8 dot products, 1 thread | 5.7 | 1.39 GB |
| Q8_0 and 2 threads | 10.6 | 1.39 GB |

Keeping weights quantized and quantizing the activations to 8 bits before each product, which is what llama.cpp does, gave about a 2.3x speedup on one thread. Its output stayed within 0.8% of the FP32 result. Adding a second thread nearly doubled it again. A Mac has more cores and more memory bandwidth, so I'd expect the threading gain to be larger there.

## What I'd change, in order

1. **Keep weights quantized** and add Q4_K and Q6_K dot products that work directly on the packed blocks. This is the biggest win for both speed and memory.
2. **Split the matrix vector rows across threads** using a persistent pool instead of creating futures for every head.
3. **Point the output layer at the embedding buffer** instead of copying it. That's an easy 1 GB saved.
4. **Write NEON kernels** for Apple Silicon.
5. **Batch the prompt.** Right now the prompt is processed one token at a time; processing it as a matrix multiply would speed up prompt reading a lot.
6. **Precompute the RoPE sine and cosine tables** instead of calling `pow`, `sin` and `cos` for every token.

## What I learned

The math in a transformer is the easy part. The speed comes from how you move bytes: keeping weights small, keeping every core busy, and not copying what you can share. Measuring first also kept me honest. My README had a memory number that was off by about five times, and I wouldn't have caught it without checking.

*The benchmark tools are in the repo: `bench/llm_bench.cpp`, `bench/matvec_bench.cpp` and `bench/compare.sh`. Raw synthetic results are in `bench/results/`.*
