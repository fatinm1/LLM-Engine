// scripts/cli_test.cpp
#include <iostream>
#include <string>
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
