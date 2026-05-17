#include "core/sampler.h"

#include "core/simd_math.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace llm {

Sampler::Sampler(const SamplerConfig& cfg) : cfg_(cfg), rng_(cfg.seed) {}

void Sampler::set_config(const SamplerConfig& cfg)
{
    cfg_ = cfg;
    rng_.seed(cfg.seed);
}

TokenID Sampler::greedy(const std::vector<float>& logits)
{
    const auto it = std::max_element(logits.begin(), logits.end());
    return static_cast<TokenID>(std::distance(logits.begin(), it));
}

TokenID Sampler::sample(std::vector<float>& logits, const std::vector<TokenID>& context)
{
    if (logits.empty()) {
        return 0;
    }

    if (cfg_.temperature <= 0.0f) {
        return greedy(logits);
    }

    const size_t ctx_start =
        context.size() > static_cast<size_t>(cfg_.repeat_last_n)
            ? context.size() - static_cast<size_t>(cfg_.repeat_last_n)
            : 0;
    for (size_t i = ctx_start; i < context.size(); ++i) {
        const TokenID tok = context[i];
        if (tok < 0 || static_cast<size_t>(tok) >= logits.size()) {
            continue;
        }
        float& logit = logits[static_cast<size_t>(tok)];
        if (logit > 0.0f) {
            logit /= cfg_.repeat_penalty;
        } else {
            logit *= cfg_.repeat_penalty;
        }
    }

    const float inv_temp = 1.0f / cfg_.temperature;
    for (float& v : logits) {
        v *= inv_temp;
    }

    if (cfg_.top_k > 0 && cfg_.top_k < static_cast<int>(logits.size())) {
        std::vector<size_t> indices(logits.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + cfg_.top_k, indices.end(),
                          [&](size_t a, size_t b) { return logits[a] > logits[b]; });
        const float threshold = logits[indices[static_cast<size_t>(cfg_.top_k - 1)]];
        for (size_t i = 0; i < logits.size(); ++i) {
            if (logits[i] < threshold) {
                logits[i] = -std::numeric_limits<float>::infinity();
            }
        }
    }

    simd::softmax(logits.data(), logits.size());

    std::vector<size_t> order;
    order.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        if (logits[i] > 0.0f) {
            order.push_back(i);
        }
    }
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return logits[a] > logits[b]; });

    float cum = 0.0f;
    size_t cutoff = order.size();
    for (size_t i = 0; i < order.size(); ++i) {
        cum += logits[order[i]];
        if (cum >= cfg_.top_p) {
            cutoff = i + 1;
            break;
        }
    }
    order.resize(cutoff);

    std::vector<float> probs(order.size());
    float sum = 0.0f;
    for (size_t i = 0; i < order.size(); ++i) {
        probs[i] = logits[order[i]];
        sum += probs[i];
    }
    std::uniform_real_distribution<float> dist(0.0f, sum);
    const float r = dist(rng_);
    float acc = 0.0f;
    for (size_t i = 0; i < order.size(); ++i) {
        acc += probs[i];
        if (r <= acc) {
            return static_cast<TokenID>(order[i]);
        }
    }
    return static_cast<TokenID>(order.back());
}

}  // namespace llm
