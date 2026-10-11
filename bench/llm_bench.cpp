// llm_bench: measures load time, prompt processing (prefill) speed, generation
// (decode) speed and peak memory for a GGUF model, so results can be compared
// with llama.cpp's llama-bench on the same file and machine.
//
// Usage:
//   llm_bench <model.gguf> [--gen N] [--reps N] [--json]
//
// Prefill and decode are timed separately. Model::generate() reports a single
// tokens/sec figure that includes prompt processing, which understates decode
// speed for short generations.

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "core/gguf_parser.h"
#include "core/model.h"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0, Clock::time_point t1)
{
    return std::chrono::duration<double>(t1 - t0).count();
}

double peak_rss_mb()
{
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
    return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0);  // bytes
#else
    return static_cast<double>(ru.ru_maxrss) / 1024.0;  // kilobytes
#endif
}

double median(std::vector<double> v)
{
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct RunResult {
    size_t prompt_tokens = 0;
    size_t gen_tokens = 0;
    double prefill_s = 0.0;  // prompt tokens plus the first generated token
    double decode_s = 0.0;   // remaining generated tokens
};

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: llm_bench <model.gguf> [--gen N] [--reps N] [--json]\n";
        return 1;
    }

    const std::string path = argv[1];
    size_t gen = 128;
    int reps = 3;
    bool json = false;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--gen") && i + 1 < argc) {
            gen = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) {
            reps = std::max(1, std::atoi(argv[++i]));
        } else if (!std::strcmp(argv[i], "--json")) {
            json = true;
        }
    }

    const auto t_load0 = Clock::now();
    auto gguf = llm::GGUFParser::parse(path);
    auto model = llm::Model::load(gguf);
    const double load_s = seconds_since(t_load0, Clock::now());

    // A prompt that tends to produce long answers, so greedy decoding rarely
    // stops early on an end of turn token.
    const std::string prompt = llm::format_llama3_prompt(
        "Write a detailed, multi-paragraph history of the printing press, "
        "from its invention to the modern day.");
    const size_t n_prompt = model->tokenizer().encode(prompt, false).size();

    llm::SamplerConfig scfg;
    scfg.temperature = 0.0f;  // greedy, deterministic

    std::vector<RunResult> runs;
    for (int r = 0; r < reps; ++r) {
        model->reset();
        RunResult rr;
        rr.prompt_tokens = n_prompt;

        Clock::time_point t_first{};
        Clock::time_point t_last{};
        const auto t0 = Clock::now();
        model->generate(prompt, scfg, gen,
                        [&](llm::TokenID, const std::string&) {
                            const auto now = Clock::now();
                            if (rr.gen_tokens == 0) {
                                t_first = now;
                            }
                            t_last = now;
                            ++rr.gen_tokens;
                            return true;
                        });
        if (rr.gen_tokens > 0) {
            rr.prefill_s = seconds_since(t0, t_first);
            rr.decode_s = seconds_since(t_first, t_last);
        }
        runs.push_back(rr);
    }

    std::vector<double> pp_tps;
    std::vector<double> tg_tps;
    for (const auto& r : runs) {
        if (r.prefill_s > 0.0) {
            pp_tps.push_back(static_cast<double>(r.prompt_tokens) / r.prefill_s);
        }
        if (r.gen_tokens > 1 && r.decode_s > 0.0) {
            tg_tps.push_back(static_cast<double>(r.gen_tokens - 1) / r.decode_s);
        }
    }

    const auto& cfg = model->config();
    const double rss = peak_rss_mb();

    if (json) {
        std::printf(
            "{\"engine\":\"llm-engine\",\"model\":\"%s\",\"layers\":%u,\"embed_dim\":%u,"
            "\"load_s\":%.3f,\"prompt_tokens\":%zu,\"gen_tokens\":%zu,\"reps\":%d,"
            "\"prefill_tok_s\":%.3f,\"decode_tok_s\":%.3f,\"peak_rss_mb\":%.1f}\n",
            path.c_str(), cfg.n_layers, cfg.embed_dim, load_s, n_prompt,
            runs.back().gen_tokens, reps, median(pp_tps), median(tg_tps), rss);
    } else {
        std::printf("model          %s\n", path.c_str());
        std::printf("layers/dim     %u / %u\n", cfg.n_layers, cfg.embed_dim);
        std::printf("load time      %.2f s\n", load_s);
        std::printf("prompt tokens  %zu\n", n_prompt);
        std::printf("gen tokens     %zu (last run)\n", runs.back().gen_tokens);
        std::printf("prefill        %.2f tok/s (median of %d)\n", median(pp_tps), reps);
        std::printf("decode         %.2f tok/s (median of %d)\n", median(tg_tps), reps);
        std::printf("peak RSS       %.0f MB\n", rss);
    }
    return 0;
}
