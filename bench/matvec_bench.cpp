// matvec_bench: estimates where decode time goes in LLM Engine using
// Llama 3.2 1B matrix shapes, and how much three changes would help:
//
//   1. fp32_1t   current design: FP32 weights, one thread per matvec
//   2. fp32_mt   FP32 weights, rows split across threads
//   3. q8_1t     Q8_0 weights kept quantized, int8 dot products, one thread
//   4. q8_mt     Q8_0 weights, rows split across threads
//
// No model file is needed. Weight values are random; speed depends only on
// shapes. Build (x86 with AVX2):
//   c++ -std=c++20 -O3 -march=native -fopenmp bench/matvec_bench.cpp -o build/matvec_bench
// On Apple Silicon, use Homebrew clang with libomp, or set OMP_NUM_THREADS=1
// and read only the single thread columns.

#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

constexpr int QK = 32;  // Q8_0 block size

struct BlockQ8 {
    float d;           // scale (fp16 on disk; fp32 here for simplicity)
    int8_t q[QK];
};

struct MatF32 {
    size_t rows, cols;
    std::vector<float> w;
};

struct MatQ8 {
    size_t rows, cols;
    std::vector<BlockQ8> b;  // rows * cols / QK blocks
};

MatF32 make_f32(size_t rows, size_t cols, std::mt19937& rng)
{
    std::normal_distribution<float> nd(0.f, 0.02f);
    MatF32 m{rows, cols, std::vector<float>(rows * cols)};
    for (auto& v : m.w) v = nd(rng);
    return m;
}

void quantize_row(const float* x, BlockQ8* out, size_t n)
{
    for (size_t b = 0; b < n / QK; ++b) {
        float amax = 0.f;
        for (int i = 0; i < QK; ++i) amax = std::max(amax, std::fabs(x[b * QK + i]));
        const float d = amax / 127.f;
        const float id = d > 0.f ? 1.f / d : 0.f;
        out[b].d = d;
        for (int i = 0; i < QK; ++i) {
            out[b].q[i] = static_cast<int8_t>(std::lround(x[b * QK + i] * id));
        }
    }
}

MatQ8 to_q8(const MatF32& m)
{
    MatQ8 q{m.rows, m.cols, std::vector<BlockQ8>(m.rows * m.cols / QK)};
    for (size_t r = 0; r < m.rows; ++r) {
        quantize_row(m.w.data() + r * m.cols, q.b.data() + r * (m.cols / QK), m.cols);
    }
    return q;
}

// Same as llm::simd::dot in src/core/simd_math.cpp
float dot_f32(const float* a, const float* b, size_t n)
{
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    }
    __m128 s = _mm_add_ps(_mm256_extractf128_ps(acc, 0), _mm256_extractf128_ps(acc, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float sum = _mm_cvtss_f32(s);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.f;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

float dot_q8(const BlockQ8* w, const BlockQ8* x, size_t nb)
{
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (size_t b = 0; b < nb; ++b) {
        const __m256i wq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w[b].q));
        const __m256i xq = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[b].q));
        const __m256i w_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(wq));
        const __m256i w_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(wq, 1));
        const __m256i x_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(xq));
        const __m256i x_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xq, 1));
        const __m256i p = _mm256_add_epi32(_mm256_madd_epi16(w_lo, x_lo),
                                           _mm256_madd_epi16(w_hi, x_hi));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(w[b].d * x[b].d), acc);
    }
    __m128 s = _mm_add_ps(_mm256_extractf128_ps(acc, 0), _mm256_extractf128_ps(acc, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
#else
    float sum = 0.f;
    for (size_t b = 0; b < nb; ++b) {
        int32_t isum = 0;
        for (int i = 0; i < QK; ++i) isum += w[b].q[i] * x[b].q[i];
        sum += static_cast<float>(isum) * w[b].d * x[b].d;
    }
    return sum;
#endif
}

void matvec_f32(const MatF32& m, const float* x, float* out, bool mt)
{
#pragma omp parallel for schedule(static) if (mt)
    for (size_t r = 0; r < m.rows; ++r) out[r] = dot_f32(m.w.data() + r * m.cols, x, m.cols);
}

void matvec_q8(const MatQ8& m, const float* x, float* out, bool mt, std::vector<BlockQ8>& xq)
{
    quantize_row(x, xq.data(), m.cols);  // activations to Q8_0, as llama.cpp does
    const size_t nb = m.cols / QK;
#pragma omp parallel for schedule(static) if (mt)
    for (size_t r = 0; r < m.rows; ++r) out[r] = dot_q8(m.b.data() + r * nb, xq.data(), nb);
}

template <class F>
double time_ms(F&& f, int iters)
{
    f();  // warm up
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) f();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

}  // namespace

int main()
{
    // Llama 3.2 1B Instruct
    const size_t D = 2048, FF = 8192, KV = 512, V = 128256, L = 16;
    std::mt19937 rng(0);

    // One transformer layer's seven matrices, plus the tied output projection.
    const size_t shapes[7][2] = {{D, D}, {KV, D}, {KV, D}, {D, D}, {FF, D}, {FF, D}, {D, FF}};
    std::vector<MatF32> layer;
    for (auto& s : shapes) layer.push_back(make_f32(s[0], s[1], rng));
    MatF32 out_proj = make_f32(V, D, rng);

    std::vector<MatQ8> layer_q;
    for (auto& m : layer) layer_q.push_back(to_q8(m));
    MatQ8 out_q = to_q8(out_proj);

    std::vector<float> x(FF, 0.01f), y(V);
    std::vector<BlockQ8> xq(FF / QK);

    auto layer_f32 = [&](bool mt) { for (auto& m : layer) matvec_f32(m, x.data(), y.data(), mt); };
    auto layer_q8 = [&](bool mt) { for (auto& m : layer_q) matvec_q8(m, x.data(), y.data(), mt, xq); };

    // Sanity check: Q8_0 result should closely match FP32 on the output projection.
    {
        std::vector<float> xr(D), a(V), b(V);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : xr) v = nd(rng);
        matvec_f32(out_proj, xr.data(), a.data(), false);
        matvec_q8(out_q, xr.data(), b.data(), false, xq);
        double num = 0, den = 0;
        for (size_t i = 0; i < V; ++i) { num += (a[i] - b[i]) * (a[i] - b[i]); den += a[i] * a[i]; }
        std::printf("q8 vs fp32 relative error: %.4f\n", std::sqrt(num / den));
    }

    const int threads = omp_get_max_threads();
    struct Row { const char* name; double layer_ms, out_ms; double bytes; };
    double layer_elems = 0;
    for (auto& s : shapes) layer_elems += static_cast<double>(s[0] * s[1]);
    const double out_elems = static_cast<double>(V * D);

    Row rows[4] = {
        {"fp32_1t", time_ms([&] { layer_f32(false); }, 5), time_ms([&] { matvec_f32(out_proj, x.data(), y.data(), false); }, 3), 4.0},
        {"fp32_mt", time_ms([&] { layer_f32(true); }, 5), time_ms([&] { matvec_f32(out_proj, x.data(), y.data(), true); }, 3), 4.0},
        {"q8_1t", time_ms([&] { layer_q8(false); }, 5), time_ms([&] { matvec_q8(out_q, x.data(), y.data(), false, xq); }, 3), 36.0 / 32.0},
        {"q8_mt", time_ms([&] { layer_q8(true); }, 5), time_ms([&] { matvec_q8(out_q, x.data(), y.data(), true, xq); }, 3), 36.0 / 32.0},
    };

    std::printf("threads available: %d\n", threads);
    std::printf("%-8s %10s %10s %12s %10s %12s\n", "variant", "layer ms", "output ms", "ms/token", "tok/s", "GB read/tok");
    for (auto& r : rows) {
        const double per_tok = L * r.layer_ms + r.out_ms;
        const double gb = (L * layer_elems + out_elems) * r.bytes / 1e9;
        std::printf("%-8s %10.2f %10.2f %12.1f %10.2f %12.2f\n", r.name, r.layer_ms, r.out_ms, per_tok, 1000.0 / per_tok, gb);
    }
    std::printf("\nms/token = 16 x layer + output projection (matvecs only; attention, norms and sampling excluded)\n");
    return 0;
}
