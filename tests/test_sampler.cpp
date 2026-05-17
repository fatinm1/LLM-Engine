#include "core/sampler.h"

#include <gtest/gtest.h>

using namespace llm;

TEST(Sampler, Greedy)
{
    const std::vector<float> logits = {0.1f, 5.0f, 0.3f};
    EXPECT_EQ(Sampler::greedy(logits), 1);
}

TEST(Sampler, TemperatureZeroIsGreedy)
{
    SamplerConfig cfg;
    cfg.temperature = 0.0f;
    Sampler sampler(cfg);
    std::vector<float> logits = {0.1f, 5.0f, 0.3f};
    const std::vector<TokenID> ctx;
    EXPECT_EQ(sampler.sample(logits, ctx), 1);
}

TEST(Sampler, RepeatPenalty)
{
    SamplerConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_k = 0;
    cfg.top_p = 1.0f;
    cfg.repeat_penalty = 2.0f;
    cfg.repeat_last_n = 8;
    cfg.seed = 1;
    Sampler sampler(cfg);

    std::vector<float> logits = {4.0f, 3.0f};
    const std::vector<TokenID> ctx = {0};
    EXPECT_EQ(sampler.sample(logits, ctx), 1);
}

TEST(Sampler, DifferentSeedsDiffer)
{
    std::vector<float> base(10, 1.0f);
    std::vector<TokenID> counts_a(10, 0);
    std::vector<TokenID> counts_b(10, 0);

    SamplerConfig cfg_a;
    cfg_a.seed = 1;
    cfg_a.top_k = 0;
    cfg_a.top_p = 1.0f;
    Sampler sampler_a(cfg_a);

    SamplerConfig cfg_b;
    cfg_b.seed = 99;
    cfg_b.top_k = 0;
    cfg_b.top_p = 1.0f;
    Sampler sampler_b(cfg_b);

    const std::vector<TokenID> ctx;
    bool differ = false;
    for (int i = 0; i < 100; ++i) {
        std::vector<float> la = base;
        std::vector<float> lb = base;
        const TokenID ta = sampler_a.sample(la, ctx);
        const TokenID tb = sampler_b.sample(lb, ctx);
        counts_a[static_cast<size_t>(ta)]++;
        counts_b[static_cast<size_t>(tb)]++;
        if (ta != tb) {
            differ = true;
        }
    }
    EXPECT_TRUE(differ);
}
