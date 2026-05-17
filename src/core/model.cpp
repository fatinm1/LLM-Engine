#include "core/gguf_parser.h"
#include "core/model.h"
#include "core/simd_math.h"
#include "core/tokenizer.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace llm {

std::unique_ptr<Model> Model::load(GGUFFile& gguf)
{
    return std::unique_ptr<Model>(new Model(gguf));
}

std::vector<float> Model::dequant_tensor(const std::string& name)
{
    const GGUFTensor* t = gguf_.find_tensor(name);
    if (!t) {
        throw std::runtime_error("Model: tensor not found: " + name);
    }

    uint64_t n_elems = 1;
    for (auto d : t->shape) {
        n_elems *= d;
    }

    std::vector<float> out(n_elems);

    switch (t->type) {
    case GGMLType::F32:
        std::memcpy(out.data(), t->data, n_elems * sizeof(float));
        break;
    case GGMLType::Q4_K_M: {
        size_t n_blocks = n_elems / 256;
        simd::dequant_q4_k_m(t->data, out.data(), n_blocks);
        break;
    }
    case GGMLType::Q8_0: {
        size_t n_blocks = n_elems / 32;
        simd::dequant_q8_0(t->data, out.data(), n_blocks);
        break;
    }
    default:
        throw std::runtime_error("Model: unsupported quant type for tensor: " + name);
    }
    return out;
}

Model::Model(GGUFFile& gguf) : gguf_(gguf)
{
    cfg_.n_layers = gguf.arch_layers();
    cfg_.n_heads = gguf.n_heads();
    cfg_.n_kv_heads = gguf.n_kv_heads();
    cfg_.embed_dim = gguf.embed_dim();
    cfg_.ff_dim = gguf.ff_dim();
    cfg_.vocab_size = gguf.vocab_size();
    cfg_.max_seq_len = gguf.context_len();
    cfg_.head_dim = cfg_.embed_dim / cfg_.n_heads;
    cfg_.rope_theta = gguf.rope_theta();

    tokenizer_ = std::make_unique<Tokenizer>(gguf);

    token_embd_ = dequant_tensor("token_embd.weight");
    output_norm_ = dequant_tensor("output_norm.weight");
    output_proj_ = dequant_tensor("output.weight");

    layer_attn_norm_.resize(cfg_.n_layers);
    layer_wq_.resize(cfg_.n_layers);
    layer_wk_.resize(cfg_.n_layers);
    layer_wv_.resize(cfg_.n_layers);
    layer_wo_.resize(cfg_.n_layers);
    layer_ffn_norm_.resize(cfg_.n_layers);
    layer_w_gate_.resize(cfg_.n_layers);
    layer_w_up_.resize(cfg_.n_layers);
    layer_w_down_.resize(cfg_.n_layers);

    for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
        std::string p = "blk." + std::to_string(L) + ".";
        layer_attn_norm_[L] = dequant_tensor(p + "attn_norm.weight");
        layer_wq_[L] = dequant_tensor(p + "attn_q.weight");
        layer_wk_[L] = dequant_tensor(p + "attn_k.weight");
        layer_wv_[L] = dequant_tensor(p + "attn_v.weight");
        layer_wo_[L] = dequant_tensor(p + "attn_output.weight");
        layer_ffn_norm_[L] = dequant_tensor(p + "ffn_norm.weight");
        layer_w_gate_[L] = dequant_tensor(p + "ffn_gate.weight");
        layer_w_up_[L] = dequant_tensor(p + "ffn_up.weight");
        layer_w_down_[L] = dequant_tensor(p + "ffn_down.weight");
    }

    const uint32_t cache_len = std::min(cfg_.max_seq_len, static_cast<uint32_t>(4096));
    kv_cache_.init(cache_len, cfg_.n_layers, cfg_.n_kv_heads, cfg_.head_dim);

    x_.resize(cfg_.embed_dim);
    x_norm_.resize(cfg_.embed_dim);
    q_.resize(cfg_.n_heads * cfg_.head_dim);
    k_.resize(cfg_.n_kv_heads * cfg_.head_dim);
    v_.resize(cfg_.n_kv_heads * cfg_.head_dim);
    attn_out_.resize(cfg_.n_heads * cfg_.head_dim);
    proj_out_.resize(cfg_.embed_dim);
    gate_.resize(cfg_.ff_dim);
    up_.resize(cfg_.ff_dim);
    ffn_out_.resize(cfg_.embed_dim);
    logits_.resize(cfg_.vocab_size);
    attn_scores_.resize(cache_len);
}

void Model::apply_rope(size_t /*pos*/) {}

void Model::attention(size_t /*L*/, size_t /*pos*/) {}

void Model::ffn(size_t /*L*/) {}

const std::vector<float>& Model::forward(TokenID /*token*/, size_t /*pos*/)
{
    return logits_;
}

}  // namespace llm
