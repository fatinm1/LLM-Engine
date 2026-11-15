// Tests for the quantized kernels and the worker pool.
//
// The key check: an integer dot product on packed weight blocks must equal the
// float dot product of the dequantized weights with the dequantized 8-bit
// activations (same math, different order of operations).

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "core/parallel.h"
#include "core/quant.h"
#include "core/simd_math.h"

using namespace llm;
using namespace llm::quant;

namespace {

std::mt19937 rng(1234);

// A random fp16 value with magnitude in roughly [2^-8, 2^-3].
uint16_t rand_fp16_scale()
{
    std::uniform_int_distribution<int> exp(7, 12), mant(0, 1023);
    return static_cast<uint16_t>((exp(rng) << 10) | mant(rng));
}

uint8_t rand_u8()
{
    return static_cast<uint8_t>(std::uniform_int_distribution<int>(0, 255)(rng));
}

std::vector<BlockQ4K> rand_q4k(size_t nb)
{
    std::vector<BlockQ4K> v(nb);
    for (auto& b : v) {
        b.d = rand_fp16_scale();
        b.dmin = rand_fp16_scale();
        for (auto& s : b.scales) s = rand_u8();
        for (auto& q : b.qs) q = rand_u8();
    }
    return v;
}

std::vector<BlockQ6K> rand_q6k(size_t nb)
{
    std::vector<BlockQ6K> v(nb);
    for (auto& b : v) {
        b.d = rand_fp16_scale();
        for (auto& q : b.ql) q = rand_u8();
        for (auto& q : b.qh) q = rand_u8();
        for (auto& s : b.scales) s = static_cast<int8_t>(rand_u8());
    }
    return v;
}

std::vector<BlockQ80> rand_q80(size_t nb)
{
    std::vector<BlockQ80> v(nb);
    for (auto& b : v) {
        b.d = rand_fp16_scale();
        for (auto& q : b.qs) q = static_cast<int8_t>(std::uniform_int_distribution<int>(-127, 127)(rng));
    }
    return v;
}

std::vector<float> rand_floats(size_t n, float scale = 1.0f)
{
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}

// Float dot of dequantized weights with dequantized Q8_K activations, in double.
double ref_dot_k(const std::vector<float>& w, const std::vector<BlockQ8K>& y)
{
    double s = 0.0;
    for (size_t i = 0; i < w.size(); ++i) {
        const BlockQ8K& b = y[i / QK_K];
        s += static_cast<double>(w[i]) * b.d * b.qs[i % QK_K];
    }
    return s;
}

double ref_dot_a(const std::vector<float>& w, const std::vector<BlockQ8A>& y)
{
    double s = 0.0;
    for (size_t i = 0; i < w.size(); ++i) {
        const BlockQ8A& b = y[i / QK8_0];
        s += static_cast<double>(w[i]) * b.d * b.qs[i % QK8_0];
    }
    return s;
}

// Relative error with a floor so values near zero don't blow up.
double rel_err(double got, double want, double scale)
{
    return std::fabs(got - want) / std::max(std::fabs(want), scale);
}

constexpr size_t kBlocks = 8;  // 2048 elements, Llama 3.2 1B's embedding size

}  // namespace

TEST(Quant, BackendIsReported)
{
    std::cout << "[ quant ] kernel backend: " << backend_name() << "\n";
    SUCCEED();
}

TEST(Quant, Q8KRoundTripAndBsums)
{
    const auto x = rand_floats(kBlocks * QK_K);
    std::vector<BlockQ8K> y(kBlocks);
    quantize_row_q8k(x.data(), y.data(), x.size());
    for (size_t b = 0; b < kBlocks; ++b) {
        for (size_t g = 0; g < 16; ++g) {
            int s = 0;
            for (size_t l = 0; l < 16; ++l) s += y[b].qs[g * 16 + l];
            EXPECT_EQ(s, y[b].bsums[g]);
        }
        for (size_t j = 0; j < QK_K; ++j) {
            EXPECT_NEAR(y[b].d * y[b].qs[j], x[b * QK_K + j], y[b].d * 0.5f + 1e-7f);
        }
    }
}

TEST(Quant, Q8KAllZeros)
{
    std::vector<float> x(QK_K, 0.0f);
    BlockQ8K y;
    quantize_row_q8k(x.data(), &y, QK_K);
    EXPECT_EQ(y.d, 0.0f);
    auto w = rand_q4k(1);
    EXPECT_EQ(vec_dot_q4k_q8k(w.data(), &y, 1), 0.0f);
}

TEST(Quant, Q4KDotMatchesDequantized)
{
    for (int trial = 0; trial < 20; ++trial) {
        auto w = rand_q4k(kBlocks);
        std::vector<float> wf(kBlocks * QK_K);
        simd::dequant_q4_k(w.data(), wf.data(), kBlocks);
        const auto x = rand_floats(kBlocks * QK_K);
        std::vector<BlockQ8K> y(kBlocks);
        quantize_row_q8k(x.data(), y.data(), x.size());

        const double want = ref_dot_k(wf, y);
        const double scale = 1e-3 * std::sqrt(static_cast<double>(wf.size()));
        EXPECT_LT(rel_err(vec_dot_q4k_q8k_ref(w.data(), y.data(), kBlocks), want, scale), 1e-4);
        EXPECT_LT(rel_err(vec_dot_q4k_q8k(w.data(), y.data(), kBlocks), want, scale), 1e-4);
    }
}

TEST(Quant, Q6KDotMatchesDequantized)
{
    for (int trial = 0; trial < 20; ++trial) {
        auto w = rand_q6k(kBlocks);
        std::vector<float> wf(kBlocks * QK_K);
        simd::dequant_q6_k(w.data(), wf.data(), kBlocks);
        const auto x = rand_floats(kBlocks * QK_K);
        std::vector<BlockQ8K> y(kBlocks);
        quantize_row_q8k(x.data(), y.data(), x.size());

        const double want = ref_dot_k(wf, y);
        const double scale = 1e-3 * std::sqrt(static_cast<double>(wf.size()));
        EXPECT_LT(rel_err(vec_dot_q6k_q8k_ref(w.data(), y.data(), kBlocks), want, scale), 1e-4);
        EXPECT_LT(rel_err(vec_dot_q6k_q8k(w.data(), y.data(), kBlocks), want, scale), 1e-4);
    }
}

TEST(Quant, Q80DotMatchesDequantized)
{
    const size_t nb = kBlocks * QK_K / QK8_0;
    for (int trial = 0; trial < 20; ++trial) {
        auto w = rand_q80(nb);
        std::vector<float> wf(nb * QK8_0);
        simd::dequant_q8_0(w.data(), wf.data(), nb);
        const auto x = rand_floats(nb * QK8_0);
        std::vector<BlockQ8A> y(nb);
        quantize_row_q8a(x.data(), y.data(), x.size());

        const double want = ref_dot_a(wf, y);
        const double scale = 1e-3 * std::sqrt(static_cast<double>(wf.size()));
        EXPECT_LT(rel_err(vec_dot_q80_q8a_ref(w.data(), y.data(), nb), want, scale), 1e-4);
        EXPECT_LT(rel_err(vec_dot_q80_q8a(w.data(), y.data(), nb), want, scale), 1e-4);
    }
}

// End to end: quantized matvec vs FP32 matvec on the dequantized weights.
// The only difference is 8-bit activation rounding, so results should agree
// to well under 1%.
TEST(Quant, MatvecCloseToFloat)
{
    const size_t rows = 64;
    const size_t cols = kBlocks * QK_K;
    auto w = rand_q4k(rows * kBlocks);

    GGUFTensor t;
    t.name = "test";
    t.shape = {cols, rows};
    t.type = GGMLType::Q4_K;
    t.data = w.data();
    t.n_bytes = w.size() * sizeof(BlockQ4K);
    t.offset = 0;

    QMatrix qm = make_matrix(t);
    QMatrix fm = make_matrix(t, /*force_f32=*/true);
    ASSERT_EQ(qm.type, GGMLType::Q4_K);
    ASSERT_EQ(fm.type, GGMLType::F32);

    const auto x = rand_floats(cols);
    QuantizedInput in;
    in.begin(x.data(), cols);
    quantize_for(in, qm.type);

    std::vector<float> got(rows), want(rows);
    matvec_rows(qm, in, got.data(), 0, rows);
    matvec_rows(fm, in, want.data(), 0, rows);

    double dot = 0, ng = 0, nw = 0;
    for (size_t r = 0; r < rows; ++r) {
        dot += static_cast<double>(got[r]) * want[r];
        ng += static_cast<double>(got[r]) * got[r];
        nw += static_cast<double>(want[r]) * want[r];
    }
    EXPECT_GT(dot / std::sqrt(ng * nw), 0.9999);
}

TEST(Quant, DequantRowMatchesFullDequant)
{
    const size_t rows = 4;
    const size_t cols = 2 * QK_K;
    auto w = rand_q6k(rows * 2);
    GGUFTensor t;
    t.name = "emb";
    t.shape = {cols, rows};
    t.type = GGMLType::Q6_K;
    t.data = w.data();
    t.n_bytes = w.size() * sizeof(BlockQ6K);
    t.offset = 0;
    QMatrix qm = make_matrix(t);
    QMatrix fm = make_matrix(t, true);
    std::vector<float> row(cols);
    for (size_t r = 0; r < rows; ++r) {
        dequant_row(qm, r, row.data());
        for (size_t c = 0; c < cols; ++c) {
            EXPECT_EQ(row[c], fm.owned[r * cols + c]);
        }
    }
}

TEST(WorkerPool, CoversEveryIndexOnce)
{
    WorkerPool pool(4);
    for (size_t n : {0u, 1u, 7u, 64u, 1000u, 4097u}) {
        for (size_t grain : {1u, 3u, 16u, 5000u}) {
            std::vector<std::atomic<int>> hits(n);
            for (auto& h : hits) h.store(0);
            pool.parallel_for(n, grain, [&](size_t b, size_t e) {
                for (size_t i = b; i < e; ++i) hits[i].fetch_add(1);
            });
            for (size_t i = 0; i < n; ++i) {
                ASSERT_EQ(hits[i].load(), 1) << "n=" << n << " grain=" << grain << " i=" << i;
            }
        }
    }
}

TEST(WorkerPool, ManyBackToBackJobs)
{
    WorkerPool pool(4);
    std::atomic<long> total{0};
    for (int job = 0; job < 2000; ++job) {
        pool.parallel_for(100, 7, [&](size_t b, size_t e) {
            total.fetch_add(static_cast<long>(e - b));
        });
    }
    EXPECT_EQ(total.load(), 2000L * 100L);
}

TEST(WorkerPool, SingleThread)
{
    WorkerPool pool(1);
    EXPECT_EQ(pool.size(), 1u);
    int calls = 0;
    pool.parallel_for(10, 2, [&](size_t b, size_t e) {
        EXPECT_EQ(b, 0u);
        EXPECT_EQ(e, 10u);
        ++calls;
    });
    EXPECT_EQ(calls, 1);
}
