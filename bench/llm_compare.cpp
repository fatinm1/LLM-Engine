// llm_compare: checks the quantized kernels against the FP32 reference path
// on a real model.
//
// Usage:
//   llm_compare <model.gguf> [--tokens N] [--threads N] [--prompt "text"]
//   llm_compare <model.gguf> --tokenize "text"   (print token ids and exit)
//
// 1. Loads the model in reference mode (all weights dequantized to FP32, the
//    engine's original math), runs the prompt, then greedy-decodes N tokens and
//    records the logits at every step.
// 2. Loads the model with quantized kernels and feeds it the exact same token
//    sequence (teacher forcing), recording its logits.
// 3. Compares them step by step: cosine similarity, whether the top-1 token
//    agrees, and the largest logit difference. Also reports how many tokens
//    the quantized model would generate on its own before diverging.
//
// The two runs differ only in 8-bit activation rounding, so a correct kernel
// agrees on nearly every top-1 token. Exits non-zero if the agreement is low.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "core/gguf_parser.h"
#include "core/model.h"
#include "core/quant.h"
#include "core/tokenizer.h"

namespace {

size_t argmax(const std::vector<float>& v)
{
    return static_cast<size_t>(std::max_element(v.begin(), v.end()) - v.begin());
}

double cosine(const std::vector<float>& a, const std::vector<float>& b)
{
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    return dot / std::sqrt(na * nb + 1e-30);
}

struct Trace {
    std::vector<llm::TokenID> tokens;           // prompt + generated
    std::vector<std::vector<float>> logits;     // one per position after the prompt's last token
};

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: llm_compare <model.gguf> [--tokens N] [--threads N] [--prompt \"text\"]\n";
        return 1;
    }
    const std::string path = argv[1];
    size_t n_gen = 64;
    llm::ModelOptions base = llm::ModelOptions::from_env();
    std::string user = "Explain how a transformer language model generates text, step by step.";
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) {
            n_gen = static_cast<size_t>(std::max(1, std::atoi(argv[++i])));
        } else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) {
            base.n_threads = static_cast<size_t>(std::max(0, std::atoi(argv[++i])));
        } else if (!std::strcmp(argv[i], "--prompt") && i + 1 < argc) {
            user = argv[++i];
        } else if (!std::strcmp(argv[i], "--tokenize") && i + 1 < argc) {
            // Tokenizer only: print ids with BOS, comma-separated, like llama-tokenize --ids.
            auto g = llm::GGUFParser::parse(path);
            llm::Tokenizer tok(g);
            const auto ids = tok.encode(argv[++i], true);
            std::printf("[");
            for (size_t k = 0; k < ids.size(); ++k) {
                std::printf("%s%u", k ? ", " : "", static_cast<unsigned>(ids[k]));
            }
            std::printf("]\n");
            return 0;
        }
    }

    auto gguf = llm::GGUFParser::parse(path);
    const std::string prompt = llm::format_llama3_prompt(user);

    // ── Reference run ──────────────────────────────────────────────────────
    Trace ref;
    {
        llm::ModelOptions o = base;
        o.reference_fp32 = true;
        std::printf("loading reference (FP32) model...\n");
        auto m = llm::Model::load(gguf, o);
        ref.tokens = m->tokenizer().encode(prompt, false);
        const size_t n_prompt = ref.tokens.size();
        for (size_t i = 0; i + 1 < n_prompt; ++i) {
            m->step(ref.tokens[i]);
        }
        llm::TokenID next = ref.tokens.back();
        for (size_t s = 0; s < n_gen; ++s) {
            const auto& lg = m->step(next);
            ref.logits.push_back(lg);
            next = static_cast<llm::TokenID>(argmax(lg));
            ref.tokens.push_back(next);
        }
        std::printf("prompt tokens: %zu, generated: %zu\n", n_prompt, n_gen);
        std::string text;
        for (size_t i = n_prompt; i < ref.tokens.size(); ++i) {
            text += m->tokenizer().decode_token(ref.tokens[i]);
        }
        std::printf("reference output: %s\n\n", text.c_str());
    }

    // ── Quantized run, teacher-forced on the reference tokens ──────────────
    llm::ModelOptions o = base;
    o.reference_fp32 = false;
    std::printf("loading quantized model (%s kernels)...\n", llm::quant::backend_name());
    auto m = llm::Model::load(gguf, o);
    const size_t n_prompt = ref.tokens.size() - n_gen;
    for (size_t i = 0; i + 1 < n_prompt; ++i) {
        m->step(ref.tokens[i]);
    }

    size_t top1_agree = 0;
    size_t first_divergence = n_gen;
    double min_cos = 1.0;
    double sum_cos = 0.0;
    double max_abs = 0.0;
    for (size_t s = 0; s < n_gen; ++s) {
        const auto& lg = m->step(ref.tokens[n_prompt - 1 + s]);
        const auto& rl = ref.logits[s];
        const double c = cosine(lg, rl);
        min_cos = std::min(min_cos, c);
        sum_cos += c;
        for (size_t i = 0; i < lg.size(); ++i) {
            max_abs = std::max(max_abs, static_cast<double>(std::fabs(lg[i] - rl[i])));
        }
        if (argmax(lg) == argmax(rl)) {
            ++top1_agree;
        } else if (first_divergence == n_gen) {
            first_divergence = s;
        }
    }

    const double agree = static_cast<double>(top1_agree) / static_cast<double>(n_gen);
    std::printf("\nsteps compared        %zu\n", n_gen);
    std::printf("top-1 agreement       %zu / %zu (%.1f%%)\n", top1_agree, n_gen, 100.0 * agree);
    std::printf("first divergence      %s\n",
                first_divergence == n_gen ? "none" : ("step " + std::to_string(first_divergence)).c_str());
    std::printf("logit cosine          mean %.6f, min %.6f\n", sum_cos / n_gen, min_cos);
    std::printf("max |logit diff|      %.4f\n", max_abs);

    const bool ok = agree >= 0.90 && min_cos >= 0.99;
    std::printf("\nresult: %s\n", ok ? "PASS" : "FAIL (quantized kernels disagree with the FP32 reference)");
    return ok ? 0 : 2;
}
