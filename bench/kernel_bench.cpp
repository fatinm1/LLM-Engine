// kernel_bench: times one matrix-vector product per weight format, using
// Llama 3.2 1B's FFN shape (8192 x 2048), single-threaded and multi-threaded.
//
// Usage: kernel_bench [--threads N] [--iters N]
//
// Reports time per product and the weight bandwidth it implies. Decode speed
// is roughly (memory bandwidth) / (weight bytes per token), so this shows how
// close each kernel gets to the memory limit.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "core/parallel.h"
#include "core/quant.h"

using namespace llm;
using namespace llm::quant;
using Clock = std::chrono::steady_clock;

namespace {

std::vector<uint8_t> random_bytes(size_t n, std::mt19937& rng)
{
    std::vector<uint8_t> v(n);
    for (auto& b : v) b = static_cast<uint8_t>(rng());
    return v;
}

// Overwrite every fp16 scale with a small finite value so the math is realistic.
void fix_scales(GGMLType type, std::vector<uint8_t>& buf)
{
    const uint16_t s = 0x2000;  // ~0.0078
    if (type == GGMLType::Q4_K) {
        for (size_t o = 0; o + sizeof(BlockQ4K) <= buf.size(); o += sizeof(BlockQ4K)) {
            std::memcpy(&buf[o], &s, 2);
            std::memcpy(&buf[o + 2], &s, 2);
        }
    } else if (type == GGMLType::Q6_K) {
        for (size_t o = 0; o + sizeof(BlockQ6K) <= buf.size(); o += sizeof(BlockQ6K)) {
            std::memcpy(&buf[o + 208], &s, 2);
        }
    } else if (type == GGMLType::Q8_0) {
        for (size_t o = 0; o + sizeof(BlockQ80) <= buf.size(); o += sizeof(BlockQ80)) {
            std::memcpy(&buf[o], &s, 2);
        }
    } else if (type == GGMLType::F32) {
        float f = 0.01f;
        for (size_t o = 0; o + 4 <= buf.size(); o += 4) std::memcpy(&buf[o], &f, 4);
    }
}

size_t bytes_per_row(GGMLType type, size_t cols)
{
    switch (type) {
    case GGMLType::Q4_K: return cols / QK_K * sizeof(BlockQ4K);
    case GGMLType::Q6_K: return cols / QK_K * sizeof(BlockQ6K);
    case GGMLType::Q8_0: return cols / QK8_0 * sizeof(BlockQ80);
    default: return cols * sizeof(float);
    }
}

}  // namespace

int main(int argc, char** argv)
{
    size_t threads = std::max(1u, std::thread::hardware_concurrency());
    int iters = 50;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::max(1, std::atoi(argv[++i]));
        if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::max(1, std::atoi(argv[++i]));
    }

    const size_t rows = 8192, cols = 2048;
    std::mt19937 rng(42);
    std::vector<float> x(cols);
    for (auto& v : x) v = std::uniform_real_distribution<float>(-1, 1)(rng);
    std::vector<float> out(rows);

    WorkerPool pool(threads);
    std::printf("kernel backend: %s, matrix %zu x %zu, %d iterations\n\n",
                backend_name(), rows, cols, iters);
    std::printf("%-6s %8s %12s %12s %12s %12s\n", "type", "MB", "1t ms", "1t GB/s",
                (std::to_string(threads) + "t ms").c_str(), (std::to_string(threads) + "t GB/s").c_str());

    for (GGMLType type : {GGMLType::F32, GGMLType::Q8_0, GGMLType::Q6_K, GGMLType::Q4_K}) {
        auto buf = random_bytes(rows * bytes_per_row(type, cols), rng);
        fix_scales(type, buf);
        GGUFTensor t;
        t.name = "bench";
        t.shape = {cols, rows};
        t.type = type;
        t.data = buf.data();
        t.n_bytes = buf.size();
        t.offset = 0;
        QMatrix W = make_matrix(t);

        QuantizedInput in;
        in.begin(x.data(), cols);
        quantize_for(in, type);

        auto time_it = [&](bool parallel) {
            std::vector<double> ms;
            for (int it = 0; it < iters + 3; ++it) {
                const auto t0 = Clock::now();
                if (parallel) {
                    const size_t grain = std::max<size_t>(4, (32 * 1024) / W.row_bytes);
                    pool.parallel_for(rows, grain, [&](size_t b, size_t e) { matvec_rows(W, in, out.data(), b, e); });
                } else {
                    matvec_rows(W, in, out.data(), 0, rows);
                }
                const double d = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                if (it >= 3) ms.push_back(d);  // skip warm-up
            }
            std::sort(ms.begin(), ms.end());
            return ms[ms.size() / 2];
        };

        const double mb = static_cast<double>(buf.size()) / (1024.0 * 1024.0);
        const double t1 = time_it(false);
        const double tn = time_it(true);
        std::printf("%-6s %8.1f %12.3f %12.1f %12.3f %12.1f\n", ggml_type_name(type).c_str(), mb,
                    t1, mb / 1024.0 / (t1 / 1000.0), tn, mb / 1024.0 / (tn / 1000.0));
    }
    return 0;
}
