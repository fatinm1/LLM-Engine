#include "core/gguf_parser.h"
#include "core/model.h"
#include "core/simd_math.h"
#include "core/tokenizer.h"

#include <algorithm>
#include <cmath>
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

void Model::apply_rope(size_t pos)
{
    auto rotate = [&](float* buf, size_t n_h) {
        for (size_t h = 0; h < n_h; ++h) {
            float* hd = buf + h * cfg_.head_dim;
            for (size_t i = 0; i < cfg_.head_dim / 2; ++i) {
                const float theta = static_cast<float>(pos) /
                                    std::pow(cfg_.rope_theta,
                                             2.0f * static_cast<float>(i) /
                                                 static_cast<float>(cfg_.head_dim));
                const float cos_t = std::cos(theta);
                const float sin_t = std::sin(theta);
                const float v0 = hd[2 * i];
                const float v1 = hd[2 * i + 1];
                hd[2 * i] = v0 * cos_t - v1 * sin_t;
                hd[2 * i + 1] = v0 * sin_t + v1 * cos_t;
            }
        }
    };
    rotate(q_.data(), cfg_.n_heads);
    rotate(k_.data(), cfg_.n_kv_heads);
}

void Model::attention(size_t L, size_t pos)
{
    const size_t D = cfg_.embed_dim;
    const size_t H = cfg_.n_heads;
    const size_t KH = cfg_.n_kv_heads;
    const size_t Dh = cfg_.head_dim;
    const size_t GQ = H / KH;
    const float sc = 1.0f / std::sqrt(static_cast<float>(Dh));

    simd::rms_norm(x_.data(), layer_attn_norm_[L].data(), x_norm_.data(), D);

    simd::matvec(layer_wq_[L].data(), x_norm_.data(), q_.data(), H * Dh, D);
    simd::matvec(layer_wk_[L].data(), x_norm_.data(), k_.data(), KH * Dh, D);
    simd::matvec(layer_wv_[L].data(), x_norm_.data(), v_.data(), KH * Dh, D);

    apply_rope(pos);

    std::memcpy(kv_cache_.key_at(L, pos), k_.data(), KH * Dh * sizeof(float));
    std::memcpy(kv_cache_.val_at(L, pos), v_.data(), KH * Dh * sizeof(float));

    const size_t seq_len = pos + 1;

    std::fill(attn_out_.begin(), attn_out_.end(), 0.0f);

    for (size_t h = 0; h < H; ++h) {
        const size_t kv_h = h / GQ;
        const float* qh = q_.data() + h * Dh;
        float* oh = attn_out_.data() + h * Dh;

        for (size_t t = 0; t < seq_len; ++t) {
            const float* kh = kv_cache_.key_at(L, t) + kv_h * Dh;
            attn_scores_[t] = simd::dot(qh, kh, Dh) * sc;
        }

        simd::softmax(attn_scores_.data(), seq_len);

        for (size_t t = 0; t < seq_len; ++t) {
            const float* vh = kv_cache_.val_at(L, t) + kv_h * Dh;
            const float w = attn_scores_[t];
            for (size_t i = 0; i < Dh; ++i) {
                oh[i] += w * vh[i];
            }
        }
    }

    simd::matvec(layer_wo_[L].data(), attn_out_.data(), proj_out_.data(), D, H * Dh);
    simd::add(x_.data(), proj_out_.data(), x_.data(), D);
}

void Model::ffn(size_t L)
{
    const size_t D = cfg_.embed_dim;
    const size_t FF = cfg_.ff_dim;

    simd::rms_norm(x_.data(), layer_ffn_norm_[L].data(), x_norm_.data(), D);

    simd::matvec(layer_w_gate_[L].data(), x_norm_.data(), gate_.data(), FF, D);
    simd::matvec(layer_w_up_[L].data(), x_norm_.data(), up_.data(), FF, D);

    simd::silu(gate_.data(), gate_.data(), FF);
    simd::mul(gate_.data(), up_.data(), gate_.data(), FF);

    simd::matvec(layer_w_down_[L].data(), gate_.data(), ffn_out_.data(), D, FF);
    simd::add(x_.data(), ffn_out_.data(), x_.data(), D);
}

const std::vector<float>& Model::forward(TokenID token, size_t pos)
{
    const size_t D = cfg_.embed_dim;

    const float* emb = token_embd_.data() + static_cast<size_t>(token) * D;
    std::memcpy(x_.data(), emb, D * sizeof(float));

    for (size_t L = 0; L < cfg_.n_layers; ++L) {
        attention(L, pos);
        ffn(L);
    }

    simd::rms_norm(x_.data(), output_norm_.data(), x_norm_.data(), D);
    simd::matvec(output_proj_.data(), x_norm_.data(), logits_.data(), cfg_.vocab_size, D);

    return logits_;
}

}  // namespace llm
