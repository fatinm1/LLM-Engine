#include "core/simd_math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llm::simd {

namespace {

float fp16_to_fp32(uint16_t h)
{
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 14;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

#if defined(__AVX2__)
float dot_avx2(const float* a, const float* b, size_t n)
{
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    __m128 lo = _mm256_extractf128_ps(acc, 0);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    float sum = _mm_cvtss_f32(sum4);
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void add_avx2(const float* a, const float* b, float* out, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

void mul_avx2(const float* a, const float* b, float* out, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
    }
    for (; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

void scale_avx2(const float* a, float s, float* out, size_t n)
{
    __m256 vs = _mm256_set1_ps(s);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vs));
    }
    for (; i < n; ++i) {
        out[i] = a[i] * s;
    }
}

float sum_sq_avx2(const float* x, size_t n)
{
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(vx, vx, acc);
    }
    __m128 lo = _mm256_extractf128_ps(acc, 0);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    float sum = _mm_cvtss_f32(sum4);
    for (; i < n; ++i) {
        sum += x[i] * x[i];
    }
    return sum;
}
#endif

float dot_scalar(const float* a, const float* b, size_t n)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

}  // namespace

float* alloc_aligned(size_t n)
{
    void* p = nullptr;
#if defined(_WIN32)
    p = _aligned_malloc(n * sizeof(float), 32);
#else
    if (posix_memalign(&p, 32, n * sizeof(float)) != 0) {
        p = nullptr;
    }
#endif
    return static_cast<float*>(p);
}

void free_aligned(float* p)
{
    if (p == nullptr) {
        return;
    }
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

void add(const float* a, const float* b, float* out, size_t n)
{
#if defined(__AVX2__)
    add_avx2(a, b, out, n);
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
#endif
}

void mul(const float* a, const float* b, float* out, size_t n)
{
#if defined(__AVX2__)
    mul_avx2(a, b, out, n);
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
#endif
}

void scale(const float* a, float s, float* out, size_t n)
{
#if defined(__AVX2__)
    scale_avx2(a, s, out, n);
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] * s;
    }
#endif
}

float dot(const float* a, const float* b, size_t n)
{
#if defined(__AVX2__)
    return dot_avx2(a, b, n);
#else
    return dot_scalar(a, b, n);
#endif
}

void matvec(const float* A, const float* x, float* out, size_t m, size_t n)
{
    for (size_t row = 0; row < m; ++row) {
        out[row] = dot(A + row * n, x, n);
    }
}

void rms_norm(const float* x, const float* w, float* out, size_t n, float eps)
{
    float sum_sq;
#if defined(__AVX2__)
    sum_sq = sum_sq_avx2(x, n);
#else
    sum_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum_sq += x[i] * x[i];
    }
#endif
    const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(n) + eps);
    for (size_t i = 0; i < n; ++i) {
        out[i] = x[i] * inv_rms * w[i];
    }
}

void silu(const float* x, float* out, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        const float v = x[i];
        out[i] = v / (1.0f + std::exp(-v));
    }
}

void softmax(float* x, size_t n)
{
    if (n == 0) {
        return;
    }
    float max_val = x[0];
    for (size_t i = 1; i < n; ++i) {
        max_val = std::max(max_val, x[i]);
    }
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    const float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < n; ++i) {
        x[i] *= inv_sum;
    }
}

void dequant_q8_0(const void* src, float* out, size_t n_blocks)
{
    const auto* blocks = static_cast<const uint8_t*>(src);
    size_t out_idx = 0;
    for (size_t b = 0; b < n_blocks; ++b) {
        const uint8_t* block = blocks + b * 34;
        float d = fp16_to_fp32(static_cast<uint16_t>(block[0]) |
                               (static_cast<uint16_t>(block[1]) << 8));
        if (!std::isfinite(d)) {
            d = 0.0f;
        }
        for (int i = 0; i < 32; ++i) {
            const int8_t q = static_cast<int8_t>(block[2 + i]);
            out[out_idx++] = static_cast<float>(q) * d;
        }
    }
}

void dequant_q3_k_m(const void* src, float* out, size_t n_blocks)
{
    // Matches ggml/src/ggml-quants.c dequantize_row_q3_K (QK_K = 256).
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const auto* blocks = static_cast<const uint8_t*>(src);
    float* y = out;

    for (size_t b = 0; b < n_blocks; ++b) {
        const uint8_t* block = blocks + b * 110;

        float d_all = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(block + 108));
        if (!std::isfinite(d_all)) {
            d_all = 0.0f;
        }

        const uint8_t* q = block + 32;
        const uint8_t* hm = block + 0;
        uint8_t m = 1;

        uint32_t aux[4];
        const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
        std::memcpy(aux, block + 96, 12);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float dl;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                dl = d_all * static_cast<float>(scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * static_cast<float>(static_cast<int8_t>(((q[l + 0] >> shift) & 3) -
                                                                        ((hm[l + 0] & m) ? 0 : 4)));
                }

                dl = d_all * static_cast<float>(scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * static_cast<float>(static_cast<int8_t>(((q[l + 16] >> shift) & 3) -
                                                                        ((hm[l + 16] & m) ? 0 : 4)));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

void dequant_q4_k_m(const void* src, float* out, size_t n_blocks)
{
    const auto* blocks = static_cast<const uint8_t*>(src);
    size_t out_idx = 0;

    for (size_t b = 0; b < n_blocks; ++b) {
        const uint8_t* block = blocks + b * 144;
        float d = fp16_to_fp32(static_cast<uint16_t>(block[0]) |
                               (static_cast<uint16_t>(block[1]) << 8));
        float dmin = fp16_to_fp32(static_cast<uint16_t>(block[2]) |
                                  (static_cast<uint16_t>(block[3]) << 8));
        if (!std::isfinite(d)) {
            d = 0.0f;
        }
        if (!std::isfinite(dmin)) {
            dmin = 0.0f;
        }

        uint8_t scales[12];
        std::memcpy(scales, block + 4, 12);

        auto get_scale = [&](int is, int shift) -> int {
            const int j = is / 2;
            const int sc = scales[j];
            if (is < 4) {
                return (sc >> shift) & 0x3F;
            }
            if (is < 8) {
                return (sc >> (shift + 2)) & 0x3F;
            }
            if (is < 10) {
                return (sc >> (shift + 4)) & 0x3F;
            }
            return (sc >> (shift + 6)) & 0x3F;
        };

        const uint8_t* qs = block + 16;
        int is = 0;
        for (int j = 0; j < 8; ++j) {
            const int sc0 = get_scale(is + 0, 0);
            const int sc1 = get_scale(is + 1, 4);
            const int sc2 = get_scale(is + 2, 0);
            const int sc3 = get_scale(is + 3, 4);
            const float dl0 = d * static_cast<float>(sc0);
            const float dl1 = d * static_cast<float>(sc1);
            const float ml0 = dmin * static_cast<float>(sc2);
            const float ml1 = dmin * static_cast<float>(sc3);

            for (int l = 0; l < 16; ++l) {
                const uint8_t byte = qs[l];
                const uint8_t lo = byte & 0x0F;
                const uint8_t hi = byte >> 4;
                out[out_idx++] = dl0 * static_cast<float>(lo) - ml0;
                out[out_idx++] = dl1 * static_cast<float>(hi) - ml1;
            }
            qs += 16;
            is += 4;
        }
    }
}

}  // namespace llm::simd
