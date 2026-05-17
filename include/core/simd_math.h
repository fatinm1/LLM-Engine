#pragma once

#include <cstddef>

namespace llm::simd {

float* alloc_aligned(size_t n);
void   free_aligned(float* p);

void  add(const float* a, const float* b, float* out, size_t n);
void  mul(const float* a, const float* b, float* out, size_t n);
void  scale(const float* a, float s, float* out, size_t n);
float dot(const float* a, const float* b, size_t n);
void  matvec(const float* A, const float* x, float* out, size_t m, size_t n);
void  rms_norm(const float* x, const float* w, float* out, size_t n, float eps = 1e-5f);
void  silu(const float* x, float* out, size_t n);
void  softmax(float* x, size_t n);
void  dequant_q3_k_m(const void* src, float* out, size_t n_blocks);
void  dequant_q4_k_m(const void* src, float* out, size_t n_blocks);
void  dequant_q8_0(const void* src, float* out, size_t n_blocks);

}  // namespace llm::simd
