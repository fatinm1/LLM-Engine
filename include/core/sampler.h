#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace llm {

using TokenID = int32_t;

struct SamplerConfig {
    float    temperature = 0.8f;
    int      top_k = 40;
    float    top_p = 0.95f;
    float    repeat_penalty = 1.1f;
    int      repeat_last_n = 64;
    uint64_t seed = 42;
};

class Sampler {
public:
    explicit Sampler(const SamplerConfig& cfg = {});
    TokenID sample(std::vector<float>& logits, const std::vector<TokenID>& context);
    static TokenID greedy(const std::vector<float>& logits);
    void set_config(const SamplerConfig& cfg);

private:
    SamplerConfig cfg_;
    std::mt19937_64 rng_;
};

}  // namespace llm
