#include "core/gguf_parser.h"
#include "core/model.h"
#include "core/simd_math.h"
#include "core/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <initializer_list>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace llm {

std::string format_llama3_prompt(
    const std::string& user_message,
    const std::string& system_message)
{
    std::string prompt;
    prompt += "<|begin_of_text|>";
    prompt += "<|start_header_id|>system<|end_header_id|>\n\n";
    prompt += system_message;
    prompt += "<|eot_id|>";
    prompt += "<|start_header_id|>user<|end_header_id|>\n\n";
    prompt += user_message;
    prompt += "<|eot_id|>";
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    return prompt;
}

ModelOptions ModelOptions::from_env()
{
    ModelOptions o;
    if (const char* t = std::getenv("LLM_THREADS")) {
        o.n_threads = static_cast<size_t>(std::max(0, std::atoi(t)));
    }
    if (const char* r = std::getenv("LLM_REFERENCE_FP32")) {
        o.reference_fp32 = r[0] != '\0' && r[0] != '0';
    }
    return o;
}

std::unique_ptr<Model> Model::load(GGUFFile& gguf)
{
    return load(gguf, ModelOptions::from_env());
}

std::unique_ptr<Model> Model::load(GGUFFile& gguf, const ModelOptions& opts)
{
    return std::unique_ptr<Model>(new Model(gguf, opts));
}

std::vector<float> Model::dequant_tensor(const std::string& name)
{
    const GGUFTensor* t = gguf_.find_tensor(name);
    if (!t) {
        throw std::runtime_error("Model: tensor not found: " + name);
    }
    quant::QMatrix m = quant::make_matrix(*t, /*force_f32=*/true);
    return std::move(m.owned);
}

quant::QMatrix Model::load_matrix(const std::string& name)
{
    const GGUFTensor* t = gguf_.find_tensor(name);
    if (!t) {
        throw std::runtime_error("Model: tensor not found: " + name);
    }
    return quant::make_matrix(*t, opts_.reference_fp32);
}

Model::Model(GGUFFile& gguf, const ModelOptions& opts) : opts_(opts), gguf_(gguf)
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

    token_embd_ = load_matrix("token_embd.weight");
    output_norm_ = dequant_tensor("output_norm.weight");
    // Llama 3.2 1B ties the output projection to the embeddings. Share the
    // matrix instead of copying it (the old code copied ~1 GB here).
    if (gguf_.find_tensor("output.weight")) {
        output_owned_ = load_matrix("output.weight");
        output_proj_ = &output_owned_;
    } else {
        output_proj_ = &token_embd_;
    }

    layer_attn_norm_.resize(cfg_.n_layers);
    layer_ffn_norm_.resize(cfg_.n_layers);
    layer_wq_.resize(cfg_.n_layers);
    layer_wk_.resize(cfg_.n_layers);
    layer_wv_.resize(cfg_.n_layers);
    layer_wo_.resize(cfg_.n_layers);
    layer_w_gate_.resize(cfg_.n_layers);
    layer_w_up_.resize(cfg_.n_layers);
    layer_w_down_.resize(cfg_.n_layers);

    for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
        std::string p = "blk." + std::to_string(L) + ".";
        layer_attn_norm_[L] = dequant_tensor(p + "attn_norm.weight");
        layer_ffn_norm_[L] = dequant_tensor(p + "ffn_norm.weight");
        layer_wq_[L] = load_matrix(p + "attn_q.weight");
        layer_wk_[L] = load_matrix(p + "attn_k.weight");
        layer_wv_[L] = load_matrix(p + "attn_v.weight");
        layer_wo_[L] = load_matrix(p + "attn_output.weight");
        layer_w_gate_[L] = load_matrix(p + "ffn_gate.weight");
        layer_w_up_[L] = load_matrix(p + "ffn_up.weight");
        layer_w_down_[L] = load_matrix(p + "ffn_down.weight");
    }

    // RoPE frequencies. Llama 3.1+ GGUFs carry per-pair scaling factors in
    // rope_freqs.weight (the "llama3" RoPE scaling); ignoring them rotates the
    // low-frequency pairs too fast at long positions.
    const size_t half = cfg_.head_dim / 2;
    rope_inv_freq_.resize(half);
    rope_cos_.resize(half);
    rope_sin_.resize(half);
    std::vector<float> freq_factors;
    if (gguf_.find_tensor("rope_freqs.weight")) {
        freq_factors = dequant_tensor("rope_freqs.weight");
        if (freq_factors.size() != half) {
            freq_factors.clear();
        }
    }
    for (size_t i = 0; i < half; ++i) {
        float inv = std::pow(cfg_.rope_theta, -2.0f * static_cast<float>(i) /
                                                  static_cast<float>(cfg_.head_dim));
        if (!freq_factors.empty()) {
            inv /= freq_factors[i];
        }
        rope_inv_freq_[i] = inv;
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
    per_head_scores_.resize(cfg_.n_heads, std::vector<float>(cache_len));

    pool_ = std::make_unique<WorkerPool>(opts_.n_threads);
}

size_t Model::weight_bytes_per_token() const
{
    size_t b = output_proj_->rows * output_proj_->row_bytes;
    for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
        for (const quant::QMatrix* m : {&layer_wq_[L], &layer_wk_[L], &layer_wv_[L], &layer_wo_[L],
                                        &layer_w_gate_[L], &layer_w_up_[L], &layer_w_down_[L]}) {
            b += m->rows * m->row_bytes;
        }
    }
    return b;
}

void Model::prepare_input(quant::QuantizedInput& in, const float* x, size_t n,
                          std::initializer_list<const quant::QMatrix*> mats)
{
    in.begin(x, n);
    for (const quant::QMatrix* m : mats) {
        quant::quantize_for(in, m->type);
    }
}

void Model::matvec(const quant::QMatrix& W, quant::QuantizedInput& in, float* out)
{
    // ~32 KB of weights per chunk keeps scheduling overhead low while leaving
    // enough chunks for dynamic load balancing across cores.
    const size_t grain = std::max<size_t>(4, (32 * 1024) / std::max<size_t>(1, W.row_bytes));
    pool_->parallel_for(W.rows, grain, [&](size_t begin, size_t end) {
        quant::matvec_rows(W, in, out, begin, end);
    });
}

void Model::apply_rope(size_t pos)
{
    const size_t half = cfg_.head_dim / 2;
    const float p = static_cast<float>(pos);
    for (size_t i = 0; i < half; ++i) {
        const float theta = p * rope_inv_freq_[i];
        rope_cos_[i] = std::cos(theta);
        rope_sin_[i] = std::sin(theta);
    }
    auto rotate = [&](float* buf, size_t n_h) {
        for (size_t h = 0; h < n_h; ++h) {
            float* hd = buf + h * cfg_.head_dim;
            for (size_t i = 0; i < half; ++i) {
                const float v0 = hd[2 * i];
                const float v1 = hd[2 * i + 1];
                hd[2 * i] = v0 * rope_cos_[i] - v1 * rope_sin_[i];
                hd[2 * i + 1] = v0 * rope_sin_[i] + v1 * rope_cos_[i];
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

    // Q, K and V read the same input: quantize it once.
    prepare_input(in_, x_norm_.data(), D, {&layer_wq_[L], &layer_wk_[L], &layer_wv_[L]});
    matvec(layer_wq_[L], in_, q_.data());
    matvec(layer_wk_[L], in_, k_.data());
    matvec(layer_wv_[L], in_, v_.data());

    apply_rope(pos);

    std::memcpy(kv_cache_.key_at(L, pos), k_.data(), KH * Dh * sizeof(float));
    std::memcpy(kv_cache_.val_at(L, pos), v_.data(), KH * Dh * sizeof(float));

    const size_t seq_len = pos + 1;

    pool_->parallel_for(H, 1, [&](size_t h_begin, size_t h_end) {
        for (size_t h = h_begin; h < h_end; ++h) {
            const size_t kv_h = h / GQ;
            const float* qh = q_.data() + h * Dh;
            float* oh = attn_out_.data() + h * Dh;
            float* scores = per_head_scores_[h].data();

            for (size_t t = 0; t < seq_len; ++t) {
                const float* kh = kv_cache_.key_at(L, t) + kv_h * Dh;
                scores[t] = simd::dot(qh, kh, Dh) * sc;
            }

            simd::softmax(scores, seq_len);

            std::fill(oh, oh + Dh, 0.0f);
            for (size_t t = 0; t < seq_len; ++t) {
                const float* vh = kv_cache_.val_at(L, t) + kv_h * Dh;
                const float w = scores[t];
                for (size_t i = 0; i < Dh; ++i) {
                    oh[i] += w * vh[i];
                }
            }
        }
    });

    prepare_input(in_, attn_out_.data(), H * Dh, {&layer_wo_[L]});
    matvec(layer_wo_[L], in_, proj_out_.data());
    simd::add(x_.data(), proj_out_.data(), x_.data(), D);
}

void Model::ffn(size_t L)
{
    const size_t D = cfg_.embed_dim;
    const size_t FF = cfg_.ff_dim;

    simd::rms_norm(x_.data(), layer_ffn_norm_[L].data(), x_norm_.data(), D);

    prepare_input(in_, x_norm_.data(), D, {&layer_w_gate_[L], &layer_w_up_[L]});
    matvec(layer_w_gate_[L], in_, gate_.data());
    matvec(layer_w_up_[L], in_, up_.data());

    simd::silu(gate_.data(), gate_.data(), FF);
    simd::mul(gate_.data(), up_.data(), gate_.data(), FF);

    prepare_input(in_, gate_.data(), FF, {&layer_w_down_[L]});
    matvec(layer_w_down_[L], in_, ffn_out_.data());
    simd::add(x_.data(), ffn_out_.data(), x_.data(), D);
}

const std::vector<float>& Model::forward(TokenID token, size_t pos)
{
    const size_t D = cfg_.embed_dim;

    if (static_cast<size_t>(token) >= token_embd_.rows) {
        throw std::runtime_error("Model::forward: token id out of range");
    }
    quant::dequant_row(token_embd_, static_cast<size_t>(token), x_.data());
    (void)D;

    for (size_t L = 0; L < cfg_.n_layers; ++L) {
        attention(L, pos);
        ffn(L);
    }

    simd::rms_norm(x_.data(), output_norm_.data(), x_norm_.data(), D);
    prepare_input(in_, x_norm_.data(), D, {output_proj_});
    matvec(*output_proj_, in_, logits_.data());

    return logits_;
}

const std::vector<float>& Model::step(TokenID token)
{
    if (kv_cache_.is_full()) {
        throw std::runtime_error("Model::step: KV cache full");
    }
    const std::vector<float>& logits = forward(token, kv_cache_.size());
    kv_cache_.advance();
    return logits;
}

std::string Model::generate(const std::string& prompt,
                            const SamplerConfig& sampler_cfg,
                            size_t max_new_tokens,
                            TokenCallback callback,
                            std::function<bool()> stop_check)
{
    Sampler sampler(sampler_cfg);
    const bool has_begin_of_text = prompt.find("<|begin_of_text|>") != std::string::npos;
    std::vector<TokenID> context = tokenizer_->encode(prompt, !has_begin_of_text);

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
        if (stop_check && stop_check()) {
            kv_cache_.clear();
            return "";
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
