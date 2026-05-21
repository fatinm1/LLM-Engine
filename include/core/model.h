#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "core/gguf_parser.h"
#include "core/kv_cache.h"
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

// Called after each generated token. Return false to stop generation.
using TokenCallback = std::function<bool(TokenID, const std::string&)>;

class Model {
public:
    // Load from a parsed GGUFFile. The GGUFFile must outlive the Model.
    static std::unique_ptr<Model> load(GGUFFile& gguf);

    // Generate a response. Streams tokens via callback if provided.
    std::string generate(
        const std::string&   prompt,
        const SamplerConfig& sampler_cfg   = {},
        size_t               max_new_tokens = 512,
        TokenCallback        callback       = nullptr
    );

    // Reset KV cache (start new conversation)
    void reset() { kv_cache_.clear(); }

    const ModelConfig& config()           const { return cfg_; }
    const Tokenizer&   tokenizer()        const { return *tokenizer_; }
    float              tokens_per_second()const { return tps_; }
    size_t             tokens_generated() const { return tokens_generated_; }

private:
    explicit Model(GGUFFile& gguf);

    // Dequantize a named tensor to a float vector
    std::vector<float> dequant_tensor(const std::string& name);

    // Single forward pass for one token at sequence position pos
    // Returns reference to internal logits_ buffer
    const std::vector<float>& forward(TokenID token, size_t pos);

    // Attention for layer L at position pos, modifies x_ in place
    void attention(size_t L, size_t pos);

    // FFN (SwiGLU) for layer L, modifies x_ in place
    void ffn(size_t L);

    // Apply RoPE to q_ and k_ for the given sequence position
    void apply_rope(size_t pos);

    // ── Config and sub-components ──────────────────────────────────────────
    ModelConfig                cfg_;
    std::unique_ptr<Tokenizer> tokenizer_;
    KVCache                    kv_cache_;
    GGUFFile&                  gguf_;

    // ── Top-level weights (dequantized at load time) ───────────────────────
    std::vector<float> token_embd_;    // [vocab_size * embed_dim]
    std::vector<float> output_norm_;   // [embed_dim]
    std::vector<float> output_proj_;   // [vocab_size * embed_dim]

    // ── Per-layer weights ─────────────────────────────────────────────────
    // Indexed as layer_attn_norm_[L], etc.
    std::vector<std::vector<float>> layer_attn_norm_;
    std::vector<std::vector<float>> layer_wq_;
    std::vector<std::vector<float>> layer_wk_;
    std::vector<std::vector<float>> layer_wv_;
    std::vector<std::vector<float>> layer_wo_;
    std::vector<std::vector<float>> layer_ffn_norm_;
    std::vector<std::vector<float>> layer_w_gate_;
    std::vector<std::vector<float>> layer_w_up_;
    std::vector<std::vector<float>> layer_w_down_;

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
    std::vector<float> attn_scores_;// attention scores [max_seq_len]

    // ── Generation stats ──────────────────────────────────────────────────
    float  tps_              = 0.f;
    size_t tokens_generated_ = 0;
};

} // namespace llm
