#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "core/gguf_parser.h"
#include "core/kv_cache.h"
#include "core/parallel.h"
#include "core/quant.h"
#include "core/sampler.h"
#include "core/tokenizer.h"

namespace llm {

struct ModelConfig {
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t embed_dim;
    uint32_t ff_dim;
    uint32_t vocab_size;
    uint32_t max_seq_len;
    uint32_t head_dim;      // embed_dim / n_heads
    float    rope_theta;
};

// Load-time options. Defaults come from environment variables so the desktop
// app and tools pick them up without code changes:
//   LLM_THREADS=N          worker threads (default: all cores)
//   LLM_REFERENCE_FP32=1   dequantize all weights to FP32 (slow; for checking
//                          the quantized kernels against a reference)
struct ModelOptions {
    size_t n_threads = 0;
    bool   reference_fp32 = false;

    static ModelOptions from_env();
};

// Called after each generated token. Return false to stop generation.
using TokenCallback = std::function<bool(TokenID, const std::string&)>;

// Wraps a user message in the Llama 3 Instruct chat template.
std::string format_llama3_prompt(
    const std::string& user_message,
    const std::string& system_message = "You are a helpful assistant.");

class Model {
public:
    // Load from a parsed GGUFFile. The GGUFFile must outlive the Model.
    static std::unique_ptr<Model> load(GGUFFile& gguf);
    static std::unique_ptr<Model> load(GGUFFile& gguf, const ModelOptions& opts);

    // Generate a response. Streams tokens via callback if provided.
    std::string generate(
        const std::string&   prompt,
        const SamplerConfig& sampler_cfg   = {},
        size_t               max_new_tokens = 512,
        TokenCallback        callback       = nullptr,
        std::function<bool()> stop_check   = nullptr
    );

    // Reset KV cache (start new conversation)
    void reset() { kv_cache_.clear(); }

    // Run one token at the next cache position and return the logits.
    // Lower-level than generate(); used by tests and benchmarks.
    const std::vector<float>& step(TokenID token);

    size_t n_threads() const { return pool_->size(); }
    bool   reference_fp32() const { return opts_.reference_fp32; }
    // Bytes of weight data read per generated token.
    size_t weight_bytes_per_token() const;

    const ModelConfig& config()           const { return cfg_; }
    const Tokenizer&   tokenizer()        const { return *tokenizer_; }
    float              tokens_per_second()const { return tps_; }
    size_t             tokens_generated() const { return tokens_generated_; }

private:
    Model(GGUFFile& gguf, const ModelOptions& opts);

    // Dequantize a named tensor to a float vector (norm weights, small tensors)
    std::vector<float> dequant_tensor(const std::string& name);

    // Weight matrix for a named tensor (zero-copy view unless conversion is needed)
    quant::QMatrix load_matrix(const std::string& name);

    // out = W * in, rows split across the worker pool. `in` must already be
    // prepared with prepare_input() for W's type.
    void matvec(const quant::QMatrix& W, quant::QuantizedInput& in, float* out);

    // Set `in` to x[0..n) and quantize it for each of the given matrices.
    static void prepare_input(quant::QuantizedInput& in, const float* x, size_t n,
                              std::initializer_list<const quant::QMatrix*> mats);

    // Single forward pass for one token at sequence position pos
    // Returns reference to internal logits_ buffer
    const std::vector<float>& forward(TokenID token, size_t pos);

    // Attention for layer L at position pos, modifies x_ in place
    void attention(size_t L, size_t pos);

    // FFN (SwiGLU) for layer L, modifies x_ in place
    void ffn(size_t L);

    // Apply RoPE to q_ and k_ for the given sequence position
    void apply_rope(size_t pos);

    ModelOptions               opts_;
    // ── Config and sub-components ──────────────────────────────────────────
    ModelConfig                cfg_;
    std::unique_ptr<Tokenizer> tokenizer_;
    KVCache                    kv_cache_;
    GGUFFile&                  gguf_;

    // ── Weights ────────────────────────────────────────────────────────────
    // Matrices stay in their GGUF block format and point into the mmap.
    // Norm weights are tiny F32 vectors.
    quant::QMatrix     token_embd_;    // [vocab_size x embed_dim]
    quant::QMatrix     output_owned_;  // separate output.weight, if the file has one
    const quant::QMatrix* output_proj_ = nullptr;  // output_owned_ or token_embd_ (tied)
    std::vector<float> output_norm_;   // [embed_dim]

    std::vector<std::vector<float>> layer_attn_norm_;
    std::vector<std::vector<float>> layer_ffn_norm_;
    std::vector<quant::QMatrix> layer_wq_;
    std::vector<quant::QMatrix> layer_wk_;
    std::vector<quant::QMatrix> layer_wv_;
    std::vector<quant::QMatrix> layer_wo_;
    std::vector<quant::QMatrix> layer_w_gate_;
    std::vector<quant::QMatrix> layer_w_up_;
    std::vector<quant::QMatrix> layer_w_down_;

    // RoPE: inverse frequency per rotated pair, with Llama 3 rope_freqs applied.
    std::vector<float> rope_inv_freq_;   // [head_dim / 2]
    std::vector<float> rope_cos_;        // scratch [head_dim / 2]
    std::vector<float> rope_sin_;        // scratch [head_dim / 2]

    // ── Working buffers (allocated once at load time) ─────────────────────
    std::vector<float> x_;          // hidden state     [embed_dim]
    std::vector<float> x_norm_;     // normalised x     [embed_dim]
    std::vector<float> q_;          // Q projection     [n_heads * head_dim]
    std::vector<float> k_;          // K projection     [n_kv_heads * head_dim]
    std::vector<float> v_;          // V projection     [n_kv_heads * head_dim]
    std::vector<float> attn_out_;   // attention output [n_heads * head_dim]
    std::vector<float> proj_out_;   // after wo         [embed_dim]
    std::vector<float> gate_;       // FFN gate         [ff_dim]
    std::vector<float> up_;         // FFN up           [ff_dim]
    std::vector<float> ffn_out_;    // after w_down     [embed_dim]
    std::vector<float> logits_;     // output logits    [vocab_size]
    std::vector<std::vector<float>> per_head_scores_; // [n_heads][max_seq_len]
    quant::QuantizedInput      in_;      // quantized activations, reused per matvec
    std::unique_ptr<WorkerPool> pool_;

    // ── Generation stats ──────────────────────────────────────────────────
    float  tps_              = 0.f;
    size_t tokens_generated_ = 0;
};

} // namespace llm
