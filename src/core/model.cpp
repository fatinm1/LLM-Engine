#include "core/gguf_parser.h"
#include "core/model.h"
#include "core/simd_math.h"
#include "core/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
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

    if (name == "blk.0.attn_q.weight") {
        const uint8_t* raw = static_cast<const uint8_t*>(t->data);
        std::cerr << "blk.0.attn_q.weight type=" << static_cast<uint32_t>(t->type)
                  << " n_elems=" << n_elems << " n_bytes=" << t->n_bytes << "\n";
        std::cerr << "First 16 raw bytes: ";
        for (int i = 0; i < 16; ++i) {
            std::cerr << std::hex << static_cast<int>(raw[i]) << " ";
        }
        std::cerr << std::dec << "\n";

        uint16_t d_raw;
        std::memcpy(&d_raw, raw + 108, 2);
        std::cerr << "d_raw (fp16) = 0x" << std::hex << d_raw << std::dec << "\n";

        std::cerr << "expected blocks=" << (n_elems / 256)
                  << " actual bytes/110=" << (t->n_bytes / 110) << "\n";

        const uint8_t* scales = raw + 96;
        uint16_t d_val;
        std::memcpy(&d_val, raw + 108, 2);
        auto fp16_to_float = [](uint16_t h) -> float {
            const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
            uint32_t exp = (h >> 10) & 0x1Fu;
            uint32_t mant = h & 0x3FFu;
            uint32_t bits;
            if (exp == 0) {
                if (mant == 0) {
                    bits = sign;
                } else {
                    exp = 127 - 14;
                    while ((mant & 0x400u) == 0) {
                        mant <<= 1;
                        --exp;
                    }
                    mant &= 0x3FFu;
                    bits = sign | (exp << 23) | (mant << 13);
                }
            } else if (exp == 31) {
                bits = sign | 0x7F800000u | (mant << 13);
            } else {
                bits = sign | ((exp + 112) << 23) | (mant << 13);
            }
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            return f;
        };
        const float d_f = fp16_to_float(d_val);

        int8_t sc[16];
        sc[0] = static_cast<int8_t>((scales[0] & 0xF) | ((scales[8] & 0x03) << 4));
        sc[1] = static_cast<int8_t>((scales[0] >> 4) | ((scales[8] & 0x0C) << 2));
        sc[2] = static_cast<int8_t>((scales[1] & 0xF) | ((scales[9] & 0x03) << 4));
        sc[3] = static_cast<int8_t>((scales[1] >> 4) | ((scales[9] & 0x0C) << 2));
        sc[4] = static_cast<int8_t>((scales[2] & 0xF) | ((scales[10] & 0x03) << 4));
        sc[5] = static_cast<int8_t>((scales[2] >> 4) | ((scales[10] & 0x0C) << 2));
        sc[6] = static_cast<int8_t>((scales[3] & 0xF) | ((scales[11] & 0x03) << 4));
        sc[7] = static_cast<int8_t>((scales[3] >> 4) | ((scales[11] & 0x0C) << 2));
        sc[8] = static_cast<int8_t>((scales[4] & 0xF) | ((scales[8] & 0x30) >> 0));
        sc[9] = static_cast<int8_t>((scales[4] >> 4) | ((scales[9] & 0x30) >> 0));
        sc[10] = static_cast<int8_t>((scales[5] & 0xF) | ((scales[10] & 0x30) >> 0));
        sc[11] = static_cast<int8_t>((scales[5] >> 4) | ((scales[11] & 0x30) >> 0));
        sc[12] = static_cast<int8_t>((scales[6] & 0xF) | ((scales[8] & 0xC0) >> 2));
        sc[13] = static_cast<int8_t>((scales[6] >> 4) | ((scales[9] & 0xC0) >> 2));
        sc[14] = static_cast<int8_t>((scales[7] & 0xF) | ((scales[10] & 0xC0) >> 2));
        sc[15] = static_cast<int8_t>((scales[7] >> 4) | ((scales[11] & 0xC0) >> 2));

        std::cerr << "d=" << d_f << " scales: ";
        for (int i = 0; i < 16; ++i) {
            const float scale_val = d_f * static_cast<float>(sc[i] - 32);
            std::cerr << "[" << i << "]sc=" << static_cast<int>(sc[i]) << " s-32="
                      << static_cast<int>(sc[i] - 32) << " scaled=" << scale_val
                      << (std::isnan(scale_val) ? " NAN!" : "") << " | ";
        }
        std::cerr << "\n";

        std::cerr << "raw scales[0..11]: ";
        for (int i = 0; i < 12; ++i) {
            std::cerr << std::hex << static_cast<int>(scales[i]) << " ";
        }
        std::cerr << std::dec << "\n";
    }

    switch (t->type) {
    case GGMLType::F32:
        std::memcpy(out.data(), t->data, n_elems * sizeof(float));
        break;
    case GGMLType::Q3_K_M:
    case GGMLType::Q3_K_L: {
        size_t n_blocks = n_elems / 256;
        simd::dequant_q3_k_m(t->data, out.data(), n_blocks);
        break;
    }
    case GGMLType::Q4_K_S:
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

    if (name == "blk.0.attn_q.weight") {
        const uint8_t* raw = static_cast<const uint8_t*>(t->data);
        int first_nan_block = -1;
        int first_nan_elem = -1;

        for (size_t i = 0; i < out.size(); ++i) {
            if (std::isnan(out[i])) {
                first_nan_block = static_cast<int>(i / 256);
                first_nan_elem = static_cast<int>(i % 256);
                break;
            }
        }

        if (first_nan_block >= 0) {
            std::cerr << "First NaN: block=" << first_nan_block << " elem=" << first_nan_elem
                      << "\n";

            const uint8_t* blk = raw + first_nan_block * 110;

            uint16_t d_raw;
            std::memcpy(&d_raw, blk + 108, 2);
            std::cerr << "d_raw=0x" << std::hex << d_raw << std::dec << "\n";

            std::cerr << "Raw block bytes: ";
            for (int i = 0; i < 110; ++i) {
                std::cerr << std::hex << static_cast<int>(blk[i]) << " ";
            }
            std::cerr << std::dec << "\n";

            int block_nans = 0;
            for (int i = 0; i < 256; ++i) {
                if (std::isnan(out[first_nan_block * 256 + i])) {
                    ++block_nans;
                }
            }
            std::cerr << "NaNs in this block: " << block_nans << "\n";
        }
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
    // Try output.weight first; fall back to token_embd.weight (weight tying)
    if (gguf_.find_tensor("output.weight")) {
        output_proj_ = dequant_tensor("output.weight");
    } else {
        output_proj_ = token_embd_;
    }

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

    {
        int nans = 0, zeros = 0;
        float mn = 1e9f, mx = -1e9f;
        for (float v : output_proj_) {
            if (std::isnan(v)) {
                ++nans;
            } else if (v == 0.f) {
                ++zeros;
            } else {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
        }
        std::cerr << "output_proj_: size=" << output_proj_.size() << " nan=" << nans << " zeros=" << zeros
                  << " min=" << mn << " max=" << mx << "\n";
    }

    {
        int nans = 0;
        float mn = 1e9f, mx = -1e9f;
        for (float v : output_norm_) {
            if (std::isnan(v)) {
                ++nans;
            } else {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
        }
        std::cerr << "output_norm_: size=" << output_norm_.size() << " nan=" << nans << " min=" << mn
                  << " max=" << mx << "\n";
    }

    auto print_layer0_stats = [](const std::vector<float>& w, const char* label) {
        int nan_count = 0, inf_count = 0, zero_count = 0;
        float min_val = std::numeric_limits<float>::max();
        float max_val = -std::numeric_limits<float>::max();
        for (float v : w) {
            if (std::isnan(v)) {
                ++nan_count;
            } else if (std::isinf(v)) {
                ++inf_count;
            } else {
                if (v == 0.f) {
                    ++zero_count;
                }
                min_val = std::min(min_val, v);
                max_val = std::max(max_val, v);
            }
        }
        std::cerr << label << " stats: size=" << w.size() << " nan=" << nan_count << " inf=" << inf_count
                  << " zeros=" << zero_count << " min=" << min_val << " max=" << max_val << "\n";
    };
    {
        int exact_zeros = 0;
        for (float v : layer_wq_[0]) {
            if (v == 0.0f) {
                ++exact_zeros;
            }
        }
        std::cerr << "layer_wq_[0] exact zeros: " << exact_zeros << " / " << layer_wq_[0].size()
                  << " (" << 100.0f * static_cast<float>(exact_zeros) /
                             static_cast<float>(layer_wq_[0].size())
                  << "%)\n";
    }
    print_layer0_stats(layer_wq_[0], "layer_wq_[0]");
    print_layer0_stats(layer_wk_[0], "layer_wk_[0]");
    print_layer0_stats(layer_attn_norm_[0], "layer_attn_norm_[0]");

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

    auto check_nan = [&](const std::vector<float>& v, const char* label) {
        for (float f : v) {
            if (std::isnan(f)) {
                std::cerr << "NaN in " << label << " layer=" << L << " pos=" << pos << "\n";
                return;
            }
        }
    };
    auto check_nan_ptr = [&](const float* p, size_t n, const char* label) {
        for (size_t i = 0; i < n; ++i) {
            if (std::isnan(p[i])) {
                std::cerr << "NaN in " << label << " layer=" << L << " pos=" << pos << "\n";
                return;
            }
        }
    };

    simd::rms_norm(x_.data(), layer_attn_norm_[L].data(), x_norm_.data(), D);
    check_nan(x_norm_, "x_norm after rms_norm");

    simd::matvec(layer_wq_[L].data(), x_norm_.data(), q_.data(), H * Dh, D);
    check_nan(q_, "q after wq matvec");

    simd::matvec(layer_wk_[L].data(), x_norm_.data(), k_.data(), KH * Dh, D);
    check_nan(k_, "k after wk matvec");

    simd::matvec(layer_wv_[L].data(), x_norm_.data(), v_.data(), KH * Dh, D);
    check_nan(v_, "v after wv matvec");

    apply_rope(pos);
    check_nan(q_, "q after rope");
    check_nan(k_, "k after rope");

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
        check_nan_ptr(attn_scores_.data(), seq_len, "attn_scores after dot");

        simd::softmax(attn_scores_.data(), seq_len);
        check_nan_ptr(attn_scores_.data(), seq_len, "attn_scores after softmax");

        for (size_t t = 0; t < seq_len; ++t) {
            const float* vh = kv_cache_.val_at(L, t) + kv_h * Dh;
            const float w = attn_scores_[t];
            for (size_t i = 0; i < Dh; ++i) {
                oh[i] += w * vh[i];
            }
        }
    }
    check_nan(attn_out_, "attn_out after weighted sum");

    simd::matvec(layer_wo_[L].data(), attn_out_.data(), proj_out_.data(), D, H * Dh);
    check_nan(proj_out_, "proj_out after wo matvec");

    simd::add(x_.data(), proj_out_.data(), x_.data(), D);
    check_nan(x_, "x after residual add");
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

        bool has_nan = false;
        for (float v : x_) {
            if (std::isnan(v)) {
                has_nan = true;
                break;
            }
        }
        if (has_nan) {
            std::cerr << "NaN in x_ after attention layer " << L << "\n";
            break;
        }

        ffn(L);

        for (float v : x_) {
            if (std::isnan(v)) {
                std::cerr << "NaN in x_ after ffn layer " << L << "\n";
                break;
            }
        }
    }

    simd::rms_norm(x_.data(), output_norm_.data(), x_norm_.data(), D);

    {
        float mag = 0.f;
        for (size_t i = 0; i < cfg_.embed_dim; ++i) {
            mag += x_norm_[i] * x_norm_[i];
        }
        mag = std::sqrt(mag / cfg_.embed_dim);
        std::cerr << "x_norm_ magnitude before output proj: " << mag << "\n";
    }

    simd::matvec(output_proj_.data(), x_norm_.data(), logits_.data(), cfg_.vocab_size, D);

    int nan_count = 0, inf_count = 0;
    for (float v : logits_) {
        if (std::isnan(v)) {
            ++nan_count;
        }
        if (std::isinf(v)) {
            ++inf_count;
        }
    }
    if (nan_count > 0 || inf_count > 0) {
        std::cerr << "forward(token=" << token << " pos=" << pos << "): nan=" << nan_count
                  << " inf=" << inf_count << "\n";
    }

    return logits_;
}

std::string Model::generate(const std::string& prompt,
                            const SamplerConfig& sampler_cfg,
                            size_t max_new_tokens,
                            TokenCallback callback)
{
    Sampler sampler(sampler_cfg);
    std::vector<TokenID> context = tokenizer_->encode(prompt, true);

    if (context.empty()) {
        throw std::runtime_error("Model::generate: empty token sequence");
    }

    std::string output;
    tokens_generated_ = 0;
    const auto t_start = std::chrono::steady_clock::now();

    for (size_t i = 0; i + 1 < context.size(); ++i) {
        if (kv_cache_.is_full()) {
            break;
        }
        forward(context[i], kv_cache_.size());
        kv_cache_.advance();
    }

    TokenID next = context.back();

    for (size_t step = 0; step < max_new_tokens; ++step) {
        if (kv_cache_.is_full()) {
            break;
        }

        const std::vector<float>& logits_ref = forward(next, kv_cache_.size());
        std::vector<float> logits(logits_ref.begin(), logits_ref.end());
        kv_cache_.advance();

        next = sampler.sample(logits, context);
        context.push_back(next);

        if (tokenizer_->is_eos(next)) {
            break;
        }

        const std::string piece = tokenizer_->decode_token(next);
        output += piece;
        ++tokens_generated_;

        const auto now = std::chrono::steady_clock::now();
        const float elapsed =
            std::chrono::duration<float>(now - t_start).count();
        tps_ = elapsed > 0.f ? static_cast<float>(tokens_generated_) / elapsed : 0.f;

        if (callback && !callback(next, piece)) {
            break;
        }
    }

    return output;
}

}  // namespace llm
