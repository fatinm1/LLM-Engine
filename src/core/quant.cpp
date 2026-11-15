#include "core/quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "core/simd_math.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llm::quant {

// ── fp16 ────────────────────────────────────────────────────────────────────

float fp16_to_fp32(uint16_t h)
{
#if defined(__ARM_NEON) && defined(__aarch64__)
    __fp16 f;
    std::memcpy(&f, &h, sizeof(f));
    return static_cast<float>(f);
#elif defined(__F16C__)
    return _cvtsh_ss(h);
#else
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
#endif
}

namespace {

// Finite-or-zero, matching the old dequant path's handling of bad scales.
inline float load_fp16(uint16_t h)
{
    const float f = fp16_to_fp32(h);
    return std::isfinite(f) ? f : 0.0f;
}

// Unpack the 6-bit scale and min for sub-block j (0..7) of a Q4_K block.
// Same as ggml's get_scale_min_k4.
inline void scale_min_k4(int j, const uint8_t* q, int& sc, int& m)
{
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

}  // namespace

// ── Activation quantization ─────────────────────────────────────────────────

void quantize_row_q8k(const float* x, BlockQ8K* y, size_t n)
{
    const size_t nb = n / QK_K;
    for (size_t i = 0; i < nb; ++i) {
        const float* xb = x + i * QK_K;
        float amax = 0.0f;
        for (size_t j = 0; j < QK_K; ++j) {
            amax = std::max(amax, std::fabs(xb[j]));
        }
        BlockQ8K& b = y[i];
        if (amax == 0.0f) {
            b.d = 0.0f;
            std::memset(b.qs, 0, sizeof(b.qs));
            std::memset(b.bsums, 0, sizeof(b.bsums));
            continue;
        }
        const float iscale = 127.0f / amax;
        for (size_t j = 0; j < QK_K; ++j) {
            const int v = static_cast<int>(std::lrintf(xb[j] * iscale));
            b.qs[j] = static_cast<int8_t>(std::clamp(v, -127, 127));
        }
        for (size_t g = 0; g < QK_K / 16; ++g) {
            int s = 0;
            for (size_t j = 0; j < 16; ++j) {
                s += b.qs[g * 16 + j];
            }
            b.bsums[g] = static_cast<int16_t>(s);
        }
        b.d = amax / 127.0f;
    }
}

void quantize_row_q8a(const float* x, BlockQ8A* y, size_t n)
{
    const size_t nb = n / QK8_0;
    for (size_t i = 0; i < nb; ++i) {
        const float* xb = x + i * QK8_0;
        float amax = 0.0f;
        for (size_t j = 0; j < QK8_0; ++j) {
            amax = std::max(amax, std::fabs(xb[j]));
        }
        BlockQ8A& b = y[i];
        const float iscale = amax > 0.0f ? 127.0f / amax : 0.0f;
        for (size_t j = 0; j < QK8_0; ++j) {
            const int v = static_cast<int>(std::lrintf(xb[j] * iscale));
            b.qs[j] = static_cast<int8_t>(std::clamp(v, -127, 127));
        }
        b.d = amax / 127.0f;
    }
}

// ── Scalar reference dot products ───────────────────────────────────────────

float vec_dot_q4k_q8k_ref(const BlockQ4K* x, const BlockQ8K* y, size_t n_blocks)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n_blocks; ++i) {
        const float d = y[i].d * load_fp16(x[i].d);
        const float dmin = y[i].d * load_fp16(x[i].dmin);
        const uint8_t* q = x[i].qs;
        const int8_t* a = y[i].qs;
        int sumi = 0;
        int summ = 0;
        for (int j = 0; j < 4; ++j) {
            int sc1, m1, sc2, m2;
            scale_min_k4(2 * j, x[i].scales, sc1, m1);
            scale_min_k4(2 * j + 1, x[i].scales, sc2, m2);
            int s1 = 0;
            int s2 = 0;
            for (int l = 0; l < 32; ++l) {
                s1 += (q[l] & 0xF) * a[l];
                s2 += (q[l] >> 4) * a[l + 32];
            }
            sumi += sc1 * s1 + sc2 * s2;
            summ += m1 * (y[i].bsums[4 * j] + y[i].bsums[4 * j + 1]) +
                    m2 * (y[i].bsums[4 * j + 2] + y[i].bsums[4 * j + 3]);
            q += 32;
            a += 64;
        }
        sum += d * static_cast<float>(sumi) - dmin * static_cast<float>(summ);
    }
    return sum;
}

float vec_dot_q6k_q8k_ref(const BlockQ6K* x, const BlockQ8K* y, size_t n_blocks)
{
    float sum = 0.0f;
    int8_t aux[QK_K];
    for (size_t i = 0; i < n_blocks; ++i) {
        // Expand to signed 6-bit values. After expansion element e uses scale e/16.
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        int8_t* o = aux;
        for (int half = 0; half < 2; ++half) {
            for (int l = 0; l < 32; ++l) {
                o[l + 0] = static_cast<int8_t>(((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                o[l + 32] = static_cast<int8_t>(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                o[l + 64] = static_cast<int8_t>(((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                o[l + 96] = static_cast<int8_t>(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            o += 128;
            ql += 64;
            qh += 32;
        }
        int sumi = 0;
        for (int g = 0; g < 16; ++g) {
            int s = 0;
            for (int l = 0; l < 16; ++l) {
                s += aux[g * 16 + l] * y[i].qs[g * 16 + l];
            }
            sumi += x[i].scales[g] * s;
        }
        sum += load_fp16(x[i].d) * y[i].d * static_cast<float>(sumi);
    }
    return sum;
}

float vec_dot_q80_q8a_ref(const BlockQ80* x, const BlockQ8A* y, size_t n_blocks)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n_blocks; ++i) {
        int s = 0;
        for (size_t l = 0; l < QK8_0; ++l) {
            s += x[i].qs[l] * y[i].qs[l];
        }
        sum += load_fp16(x[i].d) * y[i].d * static_cast<float>(s);
    }
    return sum;
}

// ── NEON kernels (Apple Silicon, ARM64 Linux) ───────────────────────────────

#if defined(__ARM_NEON) && defined(__aarch64__)
namespace {

// acc += 4 partial sums of a[i]*b[i] over 16 int8 lanes.
inline int32x4_t dot16(int32x4_t acc, int8x16_t a, int8x16_t b)
{
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(acc, a, b);
#else
    const int16x8_t p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    const int16x8_t p1 = vmull_high_s8(a, b);
    return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
#endif
}

float vec_dot_q4k_q8k_neon(const BlockQ4K* x, const BlockQ8K* y, size_t n_blocks)
{
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int32x4_t zero = vdupq_n_s32(0);
    float sum = 0.0f;
    for (size_t i = 0; i < n_blocks; ++i) {
        const float d = y[i].d * load_fp16(x[i].d);
        const float dmin = y[i].d * load_fp16(x[i].dmin);
        const uint8_t* q = x[i].qs;
        const int8_t* a = y[i].qs;
        int32_t sumi = 0;
        int32_t summ = 0;
        for (int j = 0; j < 4; ++j) {
            int sc1, m1, sc2, m2;
            scale_min_k4(2 * j, x[i].scales, sc1, m1);
            scale_min_k4(2 * j + 1, x[i].scales, sc2, m2);

            const uint8x16_t q0 = vld1q_u8(q);
            const uint8x16_t q1 = vld1q_u8(q + 16);
            const int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(q0, m4));
            const int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(q1, m4));
            const int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
            const int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));

            const int32x4_t p1 = dot16(dot16(zero, lo0, vld1q_s8(a)), lo1, vld1q_s8(a + 16));
            const int32x4_t p2 = dot16(dot16(zero, hi0, vld1q_s8(a + 32)), hi1, vld1q_s8(a + 48));
            sumi += sc1 * vaddvq_s32(p1) + sc2 * vaddvq_s32(p2);
            summ += m1 * (y[i].bsums[4 * j] + y[i].bsums[4 * j + 1]) +
                    m2 * (y[i].bsums[4 * j + 2] + y[i].bsums[4 * j + 3]);
            q += 32;
            a += 64;
        }
        sum += d * static_cast<float>(sumi) - dmin * static_cast<float>(summ);
    }
    return sum;
}

float vec_dot_q6k_q8k_neon(const BlockQ6K* x, const BlockQ8K* y, size_t n_blocks)
{
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const uint8x16_t m3 = vdupq_n_u8(0x03);
    const int8x16_t k32 = vdupq_n_s8(32);
    const int32x4_t zero = vdupq_n_s32(0);
    float sum = 0.0f;
    for (size_t i = 0; i < n_blocks; ++i) {
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;
        const int8_t* a = y[i].qs;
        int32_t sumi = 0;
        for (int half = 0; half < 2; ++half) {
            const uint8x16_t l0 = vld1q_u8(ql);
            const uint8x16_t l1 = vld1q_u8(ql + 16);
            const uint8x16_t l2 = vld1q_u8(ql + 32);
            const uint8x16_t l3 = vld1q_u8(ql + 48);
            const uint8x16_t h0 = vld1q_u8(qh);
            const uint8x16_t h1 = vld1q_u8(qh + 16);

            // Each vector is one group of 16 weights sharing one scale.
            int8x16_t v[8];
            v[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l0, m4), vshlq_n_u8(vandq_u8(h0, m3), 4)));
            v[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l1, m4), vshlq_n_u8(vandq_u8(h1, m3), 4)));
            v[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l2, m4),
                                                vshlq_n_u8(vandq_u8(vshrq_n_u8(h0, 2), m3), 4)));
            v[3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l3, m4),
                                                vshlq_n_u8(vandq_u8(vshrq_n_u8(h1, 2), m3), 4)));
            v[4] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l0, 4),
                                                vshlq_n_u8(vandq_u8(vshrq_n_u8(h0, 4), m3), 4)));
            v[5] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l1, 4),
                                                vshlq_n_u8(vandq_u8(vshrq_n_u8(h1, 4), m3), 4)));
            v[6] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l2, 4), vshlq_n_u8(vshrq_n_u8(h0, 6), 4)));
            v[7] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l3, 4), vshlq_n_u8(vshrq_n_u8(h1, 6), 4)));

            for (int g = 0; g < 8; ++g) {
                const int8x16_t w = vsubq_s8(v[g], k32);
                sumi += sc[g] * vaddvq_s32(dot16(zero, w, vld1q_s8(a + 16 * g)));
            }
            ql += 64;
            qh += 32;
            sc += 8;
            a += 128;
        }
        sum += load_fp16(x[i].d) * y[i].d * static_cast<float>(sumi);
    }
    return sum;
}

float vec_dot_q80_q8a_neon(const BlockQ80* x, const BlockQ8A* y, size_t n_blocks)
{
    const int32x4_t zero = vdupq_n_s32(0);
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (size_t i = 0; i < n_blocks; ++i) {
        const int32x4_t p = dot16(dot16(zero, vld1q_s8(x[i].qs), vld1q_s8(y[i].qs)),
                                  vld1q_s8(x[i].qs + 16), vld1q_s8(y[i].qs + 16));
        acc = vmlaq_n_f32(acc, vcvtq_f32_s32(p), load_fp16(x[i].d) * y[i].d);
    }
    return vaddvq_f32(acc);
}

}  // namespace
#endif

// ── AVX2 kernels (x86) ──────────────────────────────────────────────────────

#if defined(__AVX2__)
namespace {

inline int hsum_i32(__m256i v)
{
    const __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    const __m128i s2 = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    const __m128i s3 = _mm_add_epi32(s2, _mm_shuffle_epi32(s2, 0xB1));
    return _mm_cvtsi128_si32(s3);
}

// Signed int8 x int8 pairwise products, summed into 16 int16 lanes.
inline __m256i mul_s8_pairs(__m256i x, __m256i y)
{
    return _mm256_maddubs_epi16(_mm256_sign_epi8(x, x), _mm256_sign_epi8(y, x));
}

float vec_dot_q4k_q8k_avx2(const BlockQ4K* x, const BlockQ8K* y, size_t n_blocks)
{
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    float sum = 0.0f;
    for (size_t i = 0; i < n_blocks; ++i) {
        const float d = y[i].d * load_fp16(x[i].d);
        const float dmin = y[i].d * load_fp16(x[i].dmin);
        const uint8_t* q = x[i].qs;
        const int8_t* a = y[i].qs;
        __m256i acc = _mm256_setzero_si256();
        int32_t summ = 0;
        for (int j = 0; j < 4; ++j) {
            int sc1, m1, sc2, m2;
            scale_min_k4(2 * j, x[i].scales, sc1, m1);
            scale_min_k4(2 * j + 1, x[i].scales, sc2, m2);
            const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            const __m256i lo = _mm256_and_si256(qv, m4);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qv, 4), m4);
            const __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
            const __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 32));
            // 4-bit weights are non-negative, so maddubs (u8 x s8) applies directly.
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(lo, a0),
                                                          _mm256_set1_epi16(static_cast<int16_t>(sc1))));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(hi, a1),
                                                          _mm256_set1_epi16(static_cast<int16_t>(sc2))));
            summ += m1 * (y[i].bsums[4 * j] + y[i].bsums[4 * j + 1]) +
                    m2 * (y[i].bsums[4 * j + 2] + y[i].bsums[4 * j + 3]);
            q += 32;
            a += 64;
        }
        sum += d * static_cast<float>(hsum_i32(acc)) - dmin * static_cast<float>(summ);
    }
    return sum;
}

float vec_dot_q6k_q8k_avx2(const BlockQ6K* x, const BlockQ8K* y, size_t n_blocks)
{
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i m3 = _mm256_set1_epi8(0x03);
    const __m256i k32 = _mm256_set1_epi8(32);
    float sum = 0.0f;
    for (size_t i = 0; i < n_blocks; ++i) {
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;
        const int8_t* a = y[i].qs;
        __m256i acc = _mm256_setzero_si256();
        for (int half = 0; half < 2; ++half) {
            const __m256i l_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
            const __m256i l_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
            const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));
            __m256i v[4];
            v[0] = _mm256_or_si256(_mm256_and_si256(l_lo, m4),
                                   _mm256_slli_epi16(_mm256_and_si256(h, m3), 4));
            v[1] = _mm256_or_si256(_mm256_and_si256(l_hi, m4),
                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 2), m3), 4));
            v[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l_lo, 4), m4),
                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 4), m3), 4));
            v[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l_hi, 4), m4),
                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 6), m3), 4));
            for (int k = 0; k < 4; ++k) {
                // 32 weights: low 128-bit lane uses scale 2k, high lane scale 2k+1.
                const __m256i w = _mm256_sub_epi8(v[k], k32);
                const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 32 * k));
                const __m256i scv = _mm256_set_m128i(_mm_set1_epi16(sc[2 * k + 1]),
                                                     _mm_set1_epi16(sc[2 * k]));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mul_s8_pairs(w, av), scv));
            }
            ql += 64;
            qh += 32;
            sc += 8;
            a += 128;
        }
        sum += load_fp16(x[i].d) * y[i].d * static_cast<float>(hsum_i32(acc));
    }
    return sum;
}

float vec_dot_q80_q8a_avx2(const BlockQ80* x, const BlockQ8A* y, size_t n_blocks)
{
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (size_t i = 0; i < n_blocks; ++i) {
        const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[i].qs));
        const __m256i yv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[i].qs));
        const __m256i p = _mm256_madd_epi16(mul_s8_pairs(xv, yv), ones);
        const __m256 s = _mm256_set1_ps(load_fp16(x[i].d) * y[i].d);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(p), s));
    }
    const __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    const __m128 h2 = _mm_add_ps(h, _mm_movehl_ps(h, h));
    const __m128 h3 = _mm_add_ss(h2, _mm_shuffle_ps(h2, h2, 0x55));
    return _mm_cvtss_f32(h3);
}

}  // namespace
#endif

// ── Dispatch ────────────────────────────────────────────────────────────────

float vec_dot_q4k_q8k(const BlockQ4K* x, const BlockQ8K* y, size_t n_blocks)
{
#if defined(__ARM_NEON) && defined(__aarch64__)
    return vec_dot_q4k_q8k_neon(x, y, n_blocks);
#elif defined(__AVX2__)
    return vec_dot_q4k_q8k_avx2(x, y, n_blocks);
#else
    return vec_dot_q4k_q8k_ref(x, y, n_blocks);
#endif
}

float vec_dot_q6k_q8k(const BlockQ6K* x, const BlockQ8K* y, size_t n_blocks)
{
#if defined(__ARM_NEON) && defined(__aarch64__)
    return vec_dot_q6k_q8k_neon(x, y, n_blocks);
#elif defined(__AVX2__)
    return vec_dot_q6k_q8k_avx2(x, y, n_blocks);
#else
    return vec_dot_q6k_q8k_ref(x, y, n_blocks);
#endif
}

float vec_dot_q80_q8a(const BlockQ80* x, const BlockQ8A* y, size_t n_blocks)
{
#if defined(__ARM_NEON) && defined(__aarch64__)
    return vec_dot_q80_q8a_neon(x, y, n_blocks);
#elif defined(__AVX2__)
    return vec_dot_q80_q8a_avx2(x, y, n_blocks);
#else
    return vec_dot_q80_q8a_ref(x, y, n_blocks);
#endif
}

const char* backend_name()
{
#if defined(__ARM_NEON) && defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
    return "neon-dotprod";
#elif defined(__ARM_NEON) && defined(__aarch64__)
    return "neon";
#elif defined(__AVX2__)
    return "avx2";
#else
    return "scalar";
#endif
}

// ── Matrices ────────────────────────────────────────────────────────────────

bool is_native(GGMLType type)
{
    switch (type) {
    case GGMLType::F32:
    case GGMLType::Q4_K:
    case GGMLType::Q6_K:
    case GGMLType::Q8_0:
        return true;
    default:
        return false;
    }
}

namespace {

size_t block_elems(GGMLType type)
{
    switch (type) {
    case GGMLType::Q4_K:
    case GGMLType::Q6_K:
        return QK_K;
    case GGMLType::Q8_0:
        return QK8_0;
    default:
        return 1;
    }
}

size_t block_bytes(GGMLType type)
{
    switch (type) {
    case GGMLType::F32:
        return 4;
    case GGMLType::F16:
        return 2;
    case GGMLType::Q4_K:
        return sizeof(BlockQ4K);
    case GGMLType::Q6_K:
        return sizeof(BlockQ6K);
    case GGMLType::Q8_0:
        return sizeof(BlockQ80);
    default:
        return 0;
    }
}

// Dequantize n_elems from raw data of the given type.
void dequant_raw(GGMLType type, const void* src, float* out, size_t n_elems, const std::string& name)
{
    switch (type) {
    case GGMLType::F32:
        std::memcpy(out, src, n_elems * sizeof(float));
        break;
    case GGMLType::F16: {
        const auto* h = static_cast<const uint16_t*>(src);
        for (size_t i = 0; i < n_elems; ++i) {
            out[i] = fp16_to_fp32(h[i]);
        }
        break;
    }
    case GGMLType::Q4_K:
        simd::dequant_q4_k(src, out, n_elems / QK_K);
        break;
    case GGMLType::Q6_K:
        simd::dequant_q6_k(src, out, n_elems / QK_K);
        break;
    case GGMLType::Q8_0:
        simd::dequant_q8_0(src, out, n_elems / QK8_0);
        break;
    default:
        throw std::runtime_error("unsupported quant type " + ggml_type_name(type) +
                                 " for tensor: " + name);
    }
}

}  // namespace

QMatrix make_matrix(const GGUFTensor& t, bool force_f32)
{
    QMatrix m;
    m.cols = t.shape.empty() ? 0 : static_cast<size_t>(t.shape[0]);
    m.rows = 1;
    for (size_t i = 1; i < t.shape.size(); ++i) {
        m.rows *= static_cast<size_t>(t.shape[i]);
    }

    if (!force_f32 && is_native(t.type) && m.cols % block_elems(t.type) == 0) {
        m.type = t.type;
        m.data = static_cast<const uint8_t*>(t.data);
        m.row_bytes = (m.cols / block_elems(t.type)) * block_bytes(t.type);
        return m;
    }

    m.owned.resize(m.rows * m.cols);
    dequant_raw(t.type, t.data, m.owned.data(), m.owned.size(), t.name);
    m.type = GGMLType::F32;
    m.data = reinterpret_cast<const uint8_t*>(m.owned.data());
    m.row_bytes = m.cols * sizeof(float);
    return m;
}

void quantize_for(QuantizedInput& in, GGMLType type)
{
    switch (type) {
    case GGMLType::Q4_K:
    case GGMLType::Q6_K:
        if (!in.has_q8k) {
            in.q8k.resize(in.n / QK_K);
            quantize_row_q8k(in.f32, in.q8k.data(), in.n);
            in.has_q8k = true;
        }
        break;
    case GGMLType::Q8_0:
        if (!in.has_q8a) {
            in.q8a.resize(in.n / QK8_0);
            quantize_row_q8a(in.f32, in.q8a.data(), in.n);
            in.has_q8a = true;
        }
        break;
    default:
        break;
    }
}

void matvec_rows(const QMatrix& W, const QuantizedInput& x, float* out,
                 size_t row_begin, size_t row_end)
{
    switch (W.type) {
    case GGMLType::Q4_K: {
        const size_t nb = W.cols / QK_K;
        for (size_t r = row_begin; r < row_end; ++r) {
            const auto* row = reinterpret_cast<const BlockQ4K*>(W.data + r * W.row_bytes);
            out[r] = vec_dot_q4k_q8k(row, x.q8k.data(), nb);
        }
        break;
    }
    case GGMLType::Q6_K: {
        const size_t nb = W.cols / QK_K;
        for (size_t r = row_begin; r < row_end; ++r) {
            const auto* row = reinterpret_cast<const BlockQ6K*>(W.data + r * W.row_bytes);
            out[r] = vec_dot_q6k_q8k(row, x.q8k.data(), nb);
        }
        break;
    }
    case GGMLType::Q8_0: {
        const size_t nb = W.cols / QK8_0;
        for (size_t r = row_begin; r < row_end; ++r) {
            const auto* row = reinterpret_cast<const BlockQ80*>(W.data + r * W.row_bytes);
            out[r] = vec_dot_q80_q8a(row, x.q8a.data(), nb);
        }
        break;
    }
    case GGMLType::F32: {
        const auto* base = reinterpret_cast<const float*>(W.data);
        for (size_t r = row_begin; r < row_end; ++r) {
            out[r] = simd::dot(base + r * W.cols, x.f32, W.cols);
        }
        break;
    }
    default:
        throw std::runtime_error("matvec_rows: unsupported weight type");
    }
}

void dequant_row(const QMatrix& W, size_t row, float* out)
{
    dequant_raw(W.type, W.data + row * W.row_bytes, out, W.cols, "row");
}

}  // namespace llm::quant
