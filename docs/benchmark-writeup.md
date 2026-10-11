# Why llama.cpp generates text almost 9x faster than my from scratch LLM engine

I built [LLM Engine](https://github.com/fatinm1/LLM-Engine), a C++ inference engine that runs Llama 3.2 on a CPU with no ML libraries: my own GGUF parser, tokenizer, transformer forward pass, KV cache and sampler. It works, but llama.cpp is much faster. I wanted to know exactly why, so I measured it.

## Setup

- **Model:** Llama 3.2 1B Instruct, Q4_K_M GGUF (763 MiB on disk)
- **Machine:** MacBook Air, Apple M4, 10 cores, 24 GB RAM
- **Tools:** `llm_bench`, a small benchmark I added that times prompt processing and generation separately and records peak memory, and llama.cpp's `llama-bench` (build 11429, installed with Homebrew) on the same file
- **Settings:** a 42 token prompt, 128 generated tokens, greedy decoding, 3 runs each

One thing I found right away: the tokens per second number my app showed included the time spent reading the prompt. With a short reply, that made generation look slower than it is. `llm_bench` reports the two separately.

## Results

| engine | prompt tok/s | generation tok/s |
| --- | --- | --- |
| **LLM Engine** (mine) | **8.7** | **12.6** |
| llama.cpp, CPU, 1 thread | 50.6 | 27.5 |
| llama.cpp, CPU, 4 threads | 164.9 | 73.1 |
| llama.cpp, GPU (Metal) | 839.2 | 109.2 |

LLM Engine also peaked at **5,756 MB** of memory to run a 763 MiB file.

One oddity: my prompt speed is lower than my generation speed, even though both do the same work per token. My guess is that macOS starts the work on an efficiency core and moves it to a performance core a few seconds in, but I haven't confirmed that yet.

The fair comparison is the 1 thread row, since my engine does its heavy math on one thread. Even there, llama.cpp generates text **2.2x faster** and reads the prompt **5.8x faster**. With all its tricks turned on, it's 8.6x faster at generating and almost 100x faster at reading the prompt.

## Where the time goes

Generating one token is mostly matrix vector products: each of the 16 layers multiplies the hidden state by seven weight matrices, and then a final 128,256 by 2,048 matrix turns it into a score for every token in the vocabulary.

To check how much of the time that accounts for, I also ran the engine on a Linux machine (2 vCPU Xeon) with random weights in Llama 3.2 1B's exact shapes. The full engine took **0.413 seconds per token** there, and the matrix vector products alone, done the same way, took **0.41 seconds per token**. So nearly all of generation time is those products. Attention, normalization, RoPE and sampling barely register at this length.

Three design choices make those products slow:

**1. Every weight is converted to 32 bit floats at load.** The file stores weights in about 5 bits each. My engine expands them, so generating each token reads about **4.9 GB of weights instead of about 0.8 GB**, roughly six times more data.

At 12.6 tokens per second, that's about 62 GB/s coming out of memory, from a single thread. The M4's memory bandwidth is about 120 GB/s, so my engine is already using about half of it. That means adding threads alone could only roughly double the speed. The real limit is how many bytes each token needs.

**2. The heavy math runs on one thread.** I added a thread pool, but it only splits up the attention heads, which are tiny at this length. The matrix vector products run on a single core.

**3. The prompt is processed one token at a time.** llama.cpp processes the whole prompt as one matrix multiply, which is why its prompt speed is so far ahead even on one thread.

## The memory surprise

Expanding to 32 bit floats explains the 5,756 MB. For the 1B model, the expanded weights alone are about 6 GB:

- 3.9 GB for the 16 layers
- 1.05 GB for the token embeddings
- another 1.05 GB because the output layer shares its weights with the embeddings, but my code copies that buffer instead of pointing to it

My README said "~1.2 GB." The real number is almost five times that. llama.cpp keeps weights quantized and memory maps the file, so it stays close to the file size.

## What fixing it could look like

On the Linux machine, I benchmarked the matrix vector products four ways (median of 3 runs, matrix math only):

| variant | tok/s | weight data read per token |
| --- | --- | --- |
| current: 32 bit floats, 1 thread | 2.4 | 4.94 GB |
| 32 bit floats, 2 threads | 4.9 | 4.94 GB |
| keep weights in 8 bit (Q8_0), 1 thread | 5.7 | 1.39 GB |
| 8 bit and 2 threads | 10.6 | 1.39 GB |

Keeping weights quantized and quantizing the activations to 8 bits before each product, which is what llama.cpp does, gave about a 2.3x speedup on one thread, with results within 0.8% of the 32 bit version. Adding threads on top nearly doubled it again.

## What I'd change, in order

1. **Keep weights quantized** and write Q4_K and Q6_K dot products that work directly on the packed blocks. This is the biggest win for both speed and memory, since the M4 run shows the engine is limited by how many bytes it reads.
2. **Batch the prompt** into one matrix multiply instead of one token at a time.
3. **Split the matrix vector rows across threads** with a persistent pool, instead of creating futures for every attention head.
4. **Point the output layer at the embedding buffer** instead of copying it. That's an easy 1 GB saved.
5. **Write NEON kernels** for Apple Silicon instead of relying on the compiler.
6. **Precompute the RoPE sine and cosine tables** instead of calling `pow`, `sin` and `cos` for every token.

## What I learned

The math in a transformer is the easy part. The speed comes from how you move bytes: keeping weights small, keeping every core busy, and not copying what you can share. Measuring first also kept me honest. My README had a memory number that was off by almost five times, and I wouldn't have caught it without checking.

*The benchmark tools are in the repo: `bench/llm_bench.cpp`, `bench/matvec_bench.cpp` and `bench/compare.sh`. Raw results are in `bench/results/`.*
