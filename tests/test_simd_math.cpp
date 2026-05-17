#include "core/simd_math.h"

#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

using namespace llm::simd;

TEST(SimdMath, AllocAlignedIs32ByteAligned)
{
    float* p = alloc_aligned(64);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 32, 0u);
    free_aligned(p);
}

TEST(SimdMath, AddSmall)
{
    const float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float b[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float out[4] = {};
    add(a, b, out, 4);
    EXPECT_FLOAT_EQ(out[0], 6.0f);
    EXPECT_FLOAT_EQ(out[1], 8.0f);
    EXPECT_FLOAT_EQ(out[2], 10.0f);
    EXPECT_FLOAT_EQ(out[3], 12.0f);
}

TEST(SimdMath, AddAvxSized)
{
    std::vector<float> a(32, 1.0f);
    std::vector<float> b(32, 2.0f);
    std::vector<float> out(32, 0.0f);
    add(a.data(), b.data(), out.data(), 32);
    for (float v : out) {
        EXPECT_FLOAT_EQ(v, 3.0f);
    }
}

TEST(SimdMath, DotSmall)
{
    const float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float b[] = {1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_FLOAT_EQ(dot(a, b, 4), 30.0f);
}

TEST(SimdMath, DotLarge)
{
    std::vector<float> a(64, 1.0f);
    std::vector<float> b(64, 2.0f);
    EXPECT_FLOAT_EQ(dot(a.data(), b.data(), 64), 128.0f);
}

TEST(SimdMath, SoftmaxSumsToOne)
{
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    softmax(x.data(), x.size());
    const float sum = std::accumulate(x.begin(), x.end(), 0.0f);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(SimdMath, SoftmaxMaxGetsHighestProb)
{
    std::vector<float> x = {1.0f, 5.0f, 2.0f, 3.0f};
    softmax(x.data(), x.size());
    const auto max_it = std::max_element(x.begin(), x.end());
    EXPECT_EQ(static_cast<size_t>(max_it - x.begin()), 1u);
}

TEST(SimdMath, RmsNormUnitMagnitude)
{
    std::vector<float> x(16);
    std::vector<float> w(16, 1.0f);
    std::vector<float> out(16);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>(i + 1);
    }
    rms_norm(x.data(), w.data(), out.data(), x.size());
    float sum_sq = 0.0f;
    for (float v : out) {
        sum_sq += v * v;
    }
    const float mag = std::sqrt(sum_sq / static_cast<float>(out.size()));
    EXPECT_NEAR(mag, 1.0f, 1e-3f);
}

TEST(SimdMath, SiluZero)
{
    const float x[] = {0.0f};
    float out[1] = {-1.0f};
    silu(x, out, 1);
    EXPECT_FLOAT_EQ(out[0], 0.0f);
}

TEST(SimdMath, MatVec2x2)
{
    const float A[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    const float x[] = {1.0f, 0.0f};
    float out[2] = {};
    matvec(A, x, out, 2, 2);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 3.0f);
}

TEST(SimdMath, MulAndScale)
{
    const float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float b[] = {2.0f, 2.0f, 2.0f, 2.0f};
    float out_mul[4] = {};
    mul(a, b, out_mul, 4);
    EXPECT_FLOAT_EQ(out_mul[2], 6.0f);

    float out_scale[4] = {};
    scale(a, 3.0f, out_scale, 4);
    EXPECT_FLOAT_EQ(out_scale[1], 6.0f);
}

TEST(SimdMath, DequantQ8_0)
{
    uint8_t block[34] = {};
    block[0] = 0x00;
    block[1] = 0x3C;  // fp16 1.0
    block[2] = 1;
    block[3] = -2;
    float out[32] = {};
    dequant_q8_0(block, out, 1);
    EXPECT_NEAR(out[0], 1.0f, 1e-3f);
    EXPECT_NEAR(out[1], -2.0f, 1e-3f);
}
