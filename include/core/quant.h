#pragma once

// Quantized weight kernels.
//
// Weights stay in their GGUF block format (memory-mapped, never copied).
// Before a matrix-vector product, the input activation vector is quantized
// once to 8-bit blocks, and each output row is computed with an integer dot
// product directly on the packed weight blocks. This is the same strategy
// llama.cpp uses: decode is memory-bandwidth bound, so reading ~4.5-6.5 bits
// per weight instead of 32 is where most of the speedup comes from.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/gguf_parser.h"

namespace llm::quant {

constexpr size_t QK_K = 256;  // super-block size for K-quants
constexpr size_t QK8_0 = 32;  // block size for Q8_0

// ── On-disk weight blocks (layouts match ggml) ─────────────────────────────

#pragma pack(push, 1)
struct BlockQ4K {        // 144 bytes, 4.5 bits/weight
    uint16_t d;          // fp16 super-block scale for quantized scales
    uint16_t dmin;       // fp16 super-block scale for quantized mins
    uint8_t scales[12];  // 8 x (6-bit scale, 6-bit min), packed
    uint8_t qs[QK_K / 2];  // 4-bit quants
};
struct BlockQ6K {        // 210 bytes, 6.5625 bits/weight
    uint8_t ql[QK_K / 2];  // low 4 bits
    uint8_t qh[QK_K / 4];  // high 2 bits
    int8_t scales[QK_K / 16];  // 16 x 8-bit scales
    uint16_t d;          // fp16 super-block scale
};
struct BlockQ80 {        // 34 bytes, 8.5 bits/weight
    uint16_t d;          // fp16 scale
    int8_t qs[QK8_0];
};
#pragma pack(pop)
static_assert(sizeof(BlockQ4K) == 144);
static_assert(sizeof(BlockQ6K) == 210);
static_assert(sizeof(BlockQ80) == 34);

// ── Activation blocks (in memory only) ─────────────────────────────────────

struct BlockQ8K {        // activations for K-quant weights
    float d;             // scale: x ≈ d * q
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];  // sum of each group of 16 quants (for Q4_K mins)
};
struct BlockQ8A {        // activations for Q8_0 weights
    float d;
    int8_t qs[QK8_0];
};

// An input vector, plus the 8-bit forms the weight types that read it need.
// Call begin() with the new vector, then quantize_for() once per weight type
// that will consume it (repeat calls for the same family are free).
struct QuantizedInput {
    const float* f32 = nullptr;      // used for F32 weights
    size_t n = 0;
    std::vector<BlockQ8K> q8k;       // used for Q4_K / Q6_K weights
    std::vector<BlockQ8A> q8a;       // used for Q8_0 weights
    bool has_q8k = false;
    bool has_q8a = false;

    void begin(const float* x, size_t len)
    {
        f32 = x;
        n = len;
        has_q8k = false;
        has_q8a = false;
    }
};

// A weight matrix [rows x cols], row-major, stored in its GGUF block format.
// `data` usually points into the memory-mapped file. If the matrix had to be
// converted (unsupported on-disk type, or reference mode), `owned` holds the
// F32 copy and `data` points at it.
struct QMatrix {
    GGMLType type = GGMLType::F32;
    const uint8_t* data = nullptr;
    size_t rows = 0;
    size_t cols = 0;
    size_t row_bytes = 0;
    std::vector<float> owned;

    bool empty() const { return data == nullptr; }
};

// True if matvec() works directly on this type without conversion.
bool is_native(GGMLType type);

// Build a QMatrix view of a GGUF tensor. Types without a native kernel are
// dequantized to F32. With force_f32 every type is dequantized (reference mode).
QMatrix make_matrix(const GGUFTensor& t, bool force_f32 = false);

// Make sure `in` holds the form needed by weights of type `type`.
// in.n must be a multiple of that type's block size.
void quantize_for(QuantizedInput& in, GGMLType type);

// out[r] = dot(W[r, :], x) for r in [row_begin, row_end).
void matvec_rows(const QMatrix& W, const QuantizedInput& x, float* out,
                 size_t row_begin, size_t row_end);

// Dequantize a single row (used for embedding lookup).
void dequant_row(const QMatrix& W, size_t row, float* out);

// ── Building blocks, exposed for tests and benchmarks ──────────────────────
float fp16_to_fp32(uint16_t h);
void quantize_row_q8k(const float* x, BlockQ8K* y, size_t n);
void quantize_row_q8a(const float* x, BlockQ8A* y, size_t n);
float vec_dot_q4k_q8k(const BlockQ4K* x, const BlockQ8K* y, size_t n_blocks);
float vec_dot_q6k_q8k(const BlockQ6K* x, const BlockQ8K* y, size_t n_blocks);
float vec_dot_q80_q8a(const BlockQ80* x, const BlockQ8A* y, size_t n_blocks);

// Portable scalar versions (always compiled; used as the reference in tests).
float vec_dot_q4k_q8k_ref(const BlockQ4K* x, const BlockQ8K* y, size_t n_blocks);
float vec_dot_q6k_q8k_ref(const BlockQ6K* x, const BlockQ8K* y, size_t n_blocks);
float vec_dot_q80_q8a_ref(const BlockQ80* x, const BlockQ8A* y, size_t n_blocks);

// Name of the kernel backend compiled in ("neon-dotprod", "neon", "avx2", "scalar").
const char* backend_name();

}  // namespace llm::quant
