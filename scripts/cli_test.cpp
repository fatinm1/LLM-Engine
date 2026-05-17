// scripts/cli_test.cpp
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include "core/gguf_parser.h"
#include "core/model.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: cli_test <model.gguf>\n";
        return 1;
    }

    std::cout << "Parsing GGUF...\n" << std::flush;
    auto gguf = llm::GGUFParser::parse(argv[1]);

    std::cout << "Loading model...\n" << std::flush;
    auto model = llm::Model::load(gguf);

    const auto& cfg = model->config();
    std::cout << "Layers=" << cfg.n_layers
              << " Heads=" << cfg.n_heads
              << " KVHeads=" << cfg.n_kv_heads
              << " Dim=" << cfg.embed_dim
              << " FF=" << cfg.ff_dim
              << "\n";

    {
        std::vector<llm::TokenID> test_ctx = {128000};

        llm::SamplerConfig dbg_cfg;
        dbg_cfg.temperature = 0.f;

        int first_id = -1;
        model->generate("The", dbg_cfg, 1,
                        [&](llm::TokenID id, const std::string& piece) -> bool {
                            (void)piece;
                            first_id = id;
                            return false;
                        });
        std::cerr << "First token from 'The': id=" << first_id << " piece='"
                  << model->tokenizer().decode_token(first_id) << "'\n";
        model->reset();

        const auto& logits = model->last_logits();
        const float max_l = *std::max_element(logits.begin(), logits.end());
        const float min_l = *std::min_element(logits.begin(), logits.end());
        const size_t max_idx =
            static_cast<size_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());

        int near_max = 0;
        for (float v : logits) {
            if (max_l - v < 1.0f) {
                ++near_max;
            }
        }

        std::cerr << "Logit stats: min=" << min_l << " max=" << max_l << " argmax=" << max_idx
                  << " tokens_near_max=" << near_max << "\n";
    }

    // Test with greedy sampling (temperature=0) for determinism
    llm::SamplerConfig scfg;
    scfg.temperature = 0.0f;

    std::string prompt = "The capital of France is";
    std::cout << "\nPrompt: " << prompt << "\n";

    auto tokens = model->tokenizer().encode(prompt, true);
    std::cout << "Encoded: ";
    for (auto t : tokens) {
        std::cout << t << " ";
    }
    std::cout << "\n";

    std::cout << "Response: " << std::flush;

    int token_count = 0;
    std::string response = model->generate(
        prompt, scfg, 20,
        [&](llm::TokenID id, const std::string& piece) -> bool {
            if (token_count++ < 5) {
                std::cout << "[id=" << id << " piece='" << piece << "'] " << std::flush;
            } else {
                std::cout << piece << std::flush;
            }
            return true;
        }
    );

    std::cout << "\n\nTotal tokens: " << model->tokens_generated()
              << "\nSpeed: " << model->tokens_per_second() << " tok/s\n";

    return 0;
}
