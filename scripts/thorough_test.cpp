#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include "core/gguf_parser.h"
#include "core/model.h"

static llm::GGUFFile* g_gguf = nullptr;
static llm::Model* g_model = nullptr;

struct TestResult {
    std::string name;
    bool pass;
    std::string detail;
};

std::string greedy(const std::string& prompt, size_t max_tokens = 20)
{
    g_model->reset();
    llm::SamplerConfig cfg;
    cfg.temperature = 0.f;
    return g_model->generate(prompt, cfg, max_tokens);
}

bool contains(const std::string& haystack, const std::string& needle)
{
    std::string h = haystack;
    std::string n = needle;
    std::transform(h.begin(), h.end(), h.begin(), ::tolower);
    std::transform(n.begin(), n.end(), n.begin(), ::tolower);
    return h.find(n) != std::string::npos;
}

TestResult test_paris()
{
    std::string out = greedy("The capital of France is");
    bool pass = contains(out, "paris");
    return {"Factual: capital of France", pass, out};
}

TestResult test_sky_color()
{
    std::string out = greedy("The color of the sky is");
    bool pass = contains(out, "blue");
    return {"Factual: color of sky", pass, out};
}

TestResult test_math_simple()
{
    std::string out = greedy("2 + 2 =", 10);
    bool pass = contains(out, "4");
    return {"Math: 2 + 2", pass, out};
}

TestResult test_not_empty()
{
    std::string out = greedy("Hello", 10);
    bool pass = !out.empty();
    return {"Generation: non-empty output", pass, out};
}

TestResult test_reset_determinism()
{
    std::string out1 = greedy("The capital of France is", 10);
    std::string out2 = greedy("The capital of France is", 10);
    bool pass = (out1 == out2);
    return {"Determinism: reset gives same output", pass, "out1='" + out1 + "' out2='" + out2 + "'"};
}

TestResult test_callback_stop()
{
    g_model->reset();
    llm::SamplerConfig cfg;
    cfg.temperature = 0.f;
    size_t count = 0;
    g_model->generate(
        "Hello world", cfg, 100,
        [&](llm::TokenID, const std::string&) -> bool { return ++count < 5; });
    bool pass = (count <= 5);
    return {"Callback: stop after 5 tokens", pass, "generated " + std::to_string(count) + " tokens"};
}

TestResult test_max_tokens()
{
    g_model->reset();
    llm::SamplerConfig cfg;
    cfg.temperature = 0.f;
    g_model->generate("Tell me a very long story about", cfg, 8);
    bool pass = (g_model->tokens_generated() <= 8);
    return {"Max tokens: respects limit", pass,
            "generated " + std::to_string(g_model->tokens_generated()) + " tokens"};
}

TestResult test_speed()
{
    std::string out = greedy("The quick brown fox", 30);
    (void)out;
    float tps = g_model->tokens_per_second();
    bool pass = (tps > 0.5f);
    return {"Speed: > 0.5 tok/s", pass, std::to_string(tps) + " tok/s"};
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: thorough_test <model.gguf>\n";
        return 1;
    }

    std::cout << "Loading model...\n" << std::flush;
    auto gguf = llm::GGUFParser::parse(argv[1]);
    auto model = llm::Model::load(gguf);
    g_gguf = &gguf;
    g_model = model.get();

    std::cout << "Running tests...\n\n";

    std::vector<TestResult> results = {
        test_paris(),
        test_sky_color(),
        test_math_simple(),
        test_not_empty(),
        test_reset_determinism(),
        test_callback_stop(),
        test_max_tokens(),
        test_speed(),
    };

    int passed = 0;
    int failed = 0;
    for (const auto& r : results) {
        std::cout << (r.pass ? "PASS" : "FAIL") << "  " << r.name << "\n";
        if (!r.pass || true) {
            std::cout << "      → " << r.detail << "\n";
        }
        if (r.pass) {
            ++passed;
        } else {
            ++failed;
        }
    }

    std::cout << "\n" << passed << "/" << results.size() << " tests passed\n";
    if (failed > 0) {
        std::cout << failed << " FAILED\n";
    }

    return failed > 0 ? 1 : 0;
}
