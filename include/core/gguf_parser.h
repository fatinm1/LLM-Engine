#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace llm {

constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF"
constexpr uint32_t GGUF_VERSION = 3;

enum class GGMLType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q3_K_M = 12,
    Q3_K_L = 13,
    Q4_K_S = 14,
    Q4_K_M = 15,
    Q5_K_S = 16,
    Q5_K_M = 17,
    Q6_K = 18,
    Q8_K = 19,
    IQ2_XXS = 20,
    IQ2_XS = 21,
    IQ3_XXS = 22,
    IQ1_S = 23,
    IQ4_NL = 24,
    IQ3_S = 25,
    IQ2_S = 26,
    IQ4_XS = 27,
    IQ3_XS = 28,
    IQ1_M = 29,
    BF16 = 30,
    COUNT,
};

enum class GGUFValueType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

using GGUFScalar = std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t,
                                uint64_t, int64_t, float, double, bool, std::string>;

struct GGUFArray {
    GGUFValueType elem_type;
    std::vector<GGUFScalar> values;
};

using GGUFValue = std::variant<GGUFScalar, GGUFArray>;

struct GGUFTensor {
    std::string name;
    std::vector<uint64_t> shape;
    GGMLType type;
    uint64_t offset;
    size_t n_bytes;
    const void* data;
};

struct GGUFFile {
    uint32_t version = 0;
    uint64_t n_tensors = 0;
    uint64_t n_kv = 0;
    std::unordered_map<std::string, GGUFValue> metadata;
    std::vector<GGUFTensor> tensors;

    ~GGUFFile();

    GGUFFile(const GGUFFile&) = delete;
    GGUFFile& operator=(const GGUFFile&) = delete;
    GGUFFile(GGUFFile&&) noexcept;
    GGUFFile& operator=(GGUFFile&&) noexcept;

    const GGUFTensor* find_tensor(const std::string& name) const;

    uint32_t arch_layers() const;
    uint32_t n_heads() const;
    uint32_t n_kv_heads() const;
    uint32_t embed_dim() const;
    uint32_t ff_dim() const;
    uint32_t vocab_size() const;
    uint32_t context_len() const;
    float rope_theta() const;

private:
    friend class GGUFParser;
    GGUFFile() = default;

    void* mmap_base_ = nullptr;
    size_t mmap_size_ = 0;
#if defined(_WIN32)
    void* file_handle_ = nullptr;
    void* mapping_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

class GGUFParser {
public:
    static GGUFFile parse(const std::filesystem::path& path);
};

std::string gguf_scalar_to_string(const GGUFScalar& v);
std::string gguf_value_to_string(const GGUFValue& v);
std::string ggml_type_name(GGMLType type);

}  // namespace llm
