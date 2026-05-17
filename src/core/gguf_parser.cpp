#include "core/gguf_parser.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace llm {

namespace {

class BufferReader {
public:
    explicit BufferReader(const uint8_t* data, size_t size)
        : data_(data), size_(size) {}

    template<typename T>
    T read()
    {
        if (pos_ + sizeof(T) > size_) {
            throw std::runtime_error(
                "GGUF: unexpected end of file at offset " + std::to_string(pos_));
        }
        T value;
        std::memcpy(&value, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return value;
    }

    std::string read_string()
    {
        const uint64_t len = read<uint64_t>();
        if (pos_ + len > size_) {
            throw std::runtime_error(
                "GGUF: string length " + std::to_string(len) +
                " exceeds file at offset " + std::to_string(pos_));
        }
        std::string s(reinterpret_cast<const char*>(data_ + pos_), len);
        pos_ += len;
        return s;
    }

    size_t position() const { return pos_; }
    void seek(size_t pos) { pos_ = pos; }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_ = 0;
};

size_t ggml_type_block_size(GGMLType type)
{
    switch (type) {
    case GGMLType::Q4_K_M:
    case GGMLType::Q4_K:
    case GGMLType::Q5_K:
    case GGMLType::Q5_K_M:
    case GGMLType::Q6_K:
    case GGMLType::Q6_K_M:
    case GGMLType::Q2_K:
    case GGMLType::Q3_K:
        return 256;
    case GGMLType::Q4_0:
    case GGMLType::Q4_1:
        return 32;
    case GGMLType::Q8_0:
    case GGMLType::Q8_1:
        return 32;
    case GGMLType::F32:
    case GGMLType::F16:
    case GGMLType::BF16:
        return 1;
    default:
        return 1;
    }
}

size_t ggml_type_bytes_per_block(GGMLType type)
{
    switch (type) {
    case GGMLType::F32:
        return 4;
    case GGMLType::F16:
    case GGMLType::BF16:
        return 2;
    case GGMLType::Q4_0:
        return 18;
    case GGMLType::Q4_1:
        return 20;
    case GGMLType::Q8_0:
        return 34;
    case GGMLType::Q8_1:
        return 36;
    case GGMLType::Q4_K_M:
    case GGMLType::Q4_K:
        return 144;
    case GGMLType::Q5_K:
    case GGMLType::Q5_K_M:
        return 176;
    case GGMLType::Q6_K:
    case GGMLType::Q6_K_M:
        return 210;
    case GGMLType::Q2_K:
        return 84;
    case GGMLType::Q3_K:
        return 110;
    default:
        return 0;
    }
}

size_t tensor_n_elements(const std::vector<uint64_t>& shape)
{
    size_t n = 1;
    for (uint64_t dim : shape) {
        n *= static_cast<size_t>(dim);
    }
    return n;
}

size_t tensor_n_bytes(GGMLType type, const std::vector<uint64_t>& shape)
{
    const size_t n_el = tensor_n_elements(shape);
    const size_t block = ggml_type_block_size(type);
    const size_t bpblock = ggml_type_bytes_per_block(type);
    if (bpblock == 0) {
        return 0;
    }
    if (block == 1) {
        return n_el * bpblock;
    }
    return (n_el / block) * bpblock;
}

GGUFScalar read_scalar(BufferReader& r, GGUFValueType type)
{
    switch (type) {
    case GGUFValueType::UINT8:
        return r.read<uint8_t>();
    case GGUFValueType::INT8:
        return r.read<int8_t>();
    case GGUFValueType::UINT16:
        return r.read<uint16_t>();
    case GGUFValueType::INT16:
        return r.read<int16_t>();
    case GGUFValueType::UINT32:
        return r.read<uint32_t>();
    case GGUFValueType::INT32:
        return r.read<int32_t>();
    case GGUFValueType::UINT64:
        return r.read<uint64_t>();
    case GGUFValueType::INT64:
        return r.read<int64_t>();
    case GGUFValueType::FLOAT32:
        return r.read<float>();
    case GGUFValueType::FLOAT64:
        return r.read<double>();
    case GGUFValueType::BOOL:
        return static_cast<bool>(r.read<uint8_t>());
    case GGUFValueType::STRING:
        return r.read_string();
    default:
        throw std::runtime_error(
            "GGUF: unsupported scalar value type " + std::to_string(static_cast<uint32_t>(type)));
    }
}

GGUFValue read_value(BufferReader& r)
{
    const auto type = static_cast<GGUFValueType>(r.read<uint32_t>());
    if (type == GGUFValueType::ARRAY) {
        const auto elem_type = static_cast<GGUFValueType>(r.read<uint32_t>());
        const uint64_t count = r.read<uint64_t>();
        GGUFArray arr;
        arr.elem_type = elem_type;
        arr.values.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i) {
            arr.values.push_back(read_scalar(r, elem_type));
        }
        return arr;
    }
    return read_scalar(r, type);
}

uint32_t metadata_u32(const GGUFFile& file, const std::string& key, uint32_t def)
{
    const auto it = file.metadata.find(key);
    if (it == file.metadata.end()) {
        return def;
    }
    if (const auto* scalar = std::get_if<GGUFScalar>(&it->second)) {
        if (const auto* v = std::get_if<uint32_t>(scalar)) {
            return *v;
        }
        if (const auto* v = std::get_if<uint16_t>(scalar)) {
            return *v;
        }
        if (const auto* v = std::get_if<int32_t>(scalar)) {
            return static_cast<uint32_t>(*v);
        }
    }
    return def;
}

float metadata_f32(const GGUFFile& file, const std::string& key, float def)
{
    const auto it = file.metadata.find(key);
    if (it == file.metadata.end()) {
        return def;
    }
    if (const auto* scalar = std::get_if<GGUFScalar>(&it->second)) {
        if (const auto* v = std::get_if<float>(scalar)) {
            return *v;
        }
        if (const auto* v = std::get_if<double>(scalar)) {
            return static_cast<float>(*v);
        }
    }
    return def;
}

}  // namespace

std::string ggml_type_name(GGMLType type)
{
    switch (type) {
    case GGMLType::F32:
        return "F32";
    case GGMLType::F16:
        return "F16";
    case GGMLType::Q4_0:
        return "Q4_0";
    case GGMLType::Q8_0:
        return "Q8_0";
    case GGMLType::Q4_K_M:
        return "Q4_K_M";
    default:
        return "type_" + std::to_string(static_cast<uint32_t>(type));
    }
}

std::string gguf_scalar_to_string(const GGUFScalar& v)
{
    return std::visit(
        [](const auto& x) -> std::string {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, std::string>) {
                return x;
            }
            if constexpr (std::is_same_v<T, bool>) {
                return x ? "true" : "false";
            }
            std::ostringstream oss;
            oss << x;
            return oss.str();
        },
        v);
}

std::string gguf_value_to_string(const GGUFValue& v)
{
    if (const auto* scalar = std::get_if<GGUFScalar>(&v)) {
        return gguf_scalar_to_string(*scalar);
    }
    const auto& arr = std::get<GGUFArray>(v);
    std::ostringstream oss;
    oss << "array[" << arr.values.size() << "]";
    if (!arr.values.empty()) {
        oss << " first=" << gguf_scalar_to_string(arr.values.front());
    }
    return oss.str();
}

GGUFFile::~GGUFFile()
{
#if defined(_WIN32)
    if (mmap_base_ != nullptr) {
        UnmapViewOfFile(mmap_base_);
    }
    if (mapping_handle_ != nullptr) {
        CloseHandle(static_cast<HANDLE>(mapping_handle_));
    }
    if (file_handle_ != nullptr) {
        CloseHandle(static_cast<HANDLE>(file_handle_));
    }
#else
    if (mmap_base_ != nullptr && mmap_size_ > 0) {
        munmap(mmap_base_, mmap_size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
#endif
}

GGUFFile::GGUFFile(GGUFFile&& other) noexcept
    : version(other.version),
      n_tensors(other.n_tensors),
      n_kv(other.n_kv),
      metadata(std::move(other.metadata)),
      tensors(std::move(other.tensors)),
      mmap_base_(other.mmap_base_),
      mmap_size_(other.mmap_size_)
#if defined(_WIN32)
      ,
      file_handle_(other.file_handle_),
      mapping_handle_(other.mapping_handle_)
#else
      ,
      fd_(other.fd_)
#endif
{
    other.mmap_base_ = nullptr;
    other.mmap_size_ = 0;
#if defined(_WIN32)
    other.file_handle_ = nullptr;
    other.mapping_handle_ = nullptr;
#else
    other.fd_ = -1;
#endif
}

GGUFFile& GGUFFile::operator=(GGUFFile&& other) noexcept
{
    if (this != &other) {
        this->~GGUFFile();
        version = other.version;
        n_tensors = other.n_tensors;
        n_kv = other.n_kv;
        metadata = std::move(other.metadata);
        tensors = std::move(other.tensors);
        mmap_base_ = other.mmap_base_;
        mmap_size_ = other.mmap_size_;
#if defined(_WIN32)
        file_handle_ = other.file_handle_;
        mapping_handle_ = other.mapping_handle_;
        other.file_handle_ = nullptr;
        other.mapping_handle_ = nullptr;
#else
        fd_ = other.fd_;
        other.fd_ = -1;
#endif
        other.mmap_base_ = nullptr;
        other.mmap_size_ = 0;
    }
    return *this;
}

const GGUFTensor* GGUFFile::find_tensor(const std::string& name) const
{
    for (const auto& t : tensors) {
        if (t.name == name) {
            return &t;
        }
    }
    return nullptr;
}

uint32_t GGUFFile::arch_layers() const
{
    return metadata_u32(*this, "llama.block_count", metadata_u32(*this, "general.block_count", 0));
}

uint32_t GGUFFile::n_heads() const
{
    return metadata_u32(*this, "llama.attention.head_count", 0);
}

uint32_t GGUFFile::n_kv_heads() const
{
    const uint32_t kv = metadata_u32(*this, "llama.attention.head_count_kv", 0);
    return kv > 0 ? kv : n_heads();
}

uint32_t GGUFFile::embed_dim() const
{
    return metadata_u32(*this, "llama.embedding_length", 0);
}

uint32_t GGUFFile::ff_dim() const
{
    return metadata_u32(*this, "llama.feed_forward_length", 0);
}

uint32_t GGUFFile::vocab_size() const
{
    return metadata_u32(*this, "llama.vocab_size",
                        metadata_u32(*this, "tokenizer.ggml.n_vocab", 0));
}

uint32_t GGUFFile::context_len() const
{
    return metadata_u32(*this, "llama.context_length", 2048);
}

float GGUFFile::rope_theta() const
{
    return metadata_f32(*this, "llama.rope.freq_base", 10000.0f);
}

GGUFFile GGUFParser::parse(const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("GGUF: file not found: " + path.string());
    }

    GGUFFile file;

#if defined(_WIN32)
    const std::wstring wpath = path.wstring();
    HANDLE fh = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("GGUF: failed to open file: " + path.string());
    }
    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(fh, &file_size)) {
        CloseHandle(fh);
        throw std::runtime_error("GGUF: failed to stat file: " + path.string());
    }
    HANDLE mh = CreateFileMappingW(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mh == nullptr) {
        CloseHandle(fh);
        throw std::runtime_error("GGUF: failed to create file mapping: " + path.string());
    }
    void* base = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (base == nullptr) {
        CloseHandle(mh);
        CloseHandle(fh);
        throw std::runtime_error("GGUF: failed to map file: " + path.string());
    }
    file.file_handle_ = fh;
    file.mapping_handle_ = mh;
    file.mmap_base_ = base;
    file.mmap_size_ = static_cast<size_t>(file_size.QuadPart);
#else
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("GGUF: failed to open file: " + path.string());
    }
    struct stat st {};
    if (fstat(fd, &st) != 0) {
        close(fd);
        throw std::runtime_error("GGUF: failed to stat file: " + path.string());
    }
    if (st.st_size <= 0) {
        close(fd);
        throw std::runtime_error("GGUF: empty file: " + path.string());
    }
    void* base = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("GGUF: mmap failed for file: " + path.string());
    }
#if defined(MADV_SEQUENTIAL)
    madvise(base, static_cast<size_t>(st.st_size), MADV_SEQUENTIAL);
#endif
    file.fd_ = fd;
    file.mmap_base_ = base;
    file.mmap_size_ = static_cast<size_t>(st.st_size);
#endif

    const auto* bytes = static_cast<const uint8_t*>(file.mmap_base_);
    BufferReader reader(bytes, file.mmap_size_);

    const uint32_t magic = reader.read<uint32_t>();
    if (magic != GGUF_MAGIC) {
        throw std::runtime_error(
            "GGUF: invalid magic 0x" + std::to_string(magic) + " in file " + path.string());
    }

    file.version = reader.read<uint32_t>();
    if (file.version > GGUF_VERSION) {
        throw std::runtime_error(
            "GGUF: unsupported version " + std::to_string(file.version) + " in file " +
            path.string());
    }

    file.n_tensors = reader.read<uint64_t>();
    file.n_kv = reader.read<uint64_t>();

    for (uint64_t i = 0; i < file.n_kv; ++i) {
        const std::string key = reader.read_string();
        GGUFValue value = read_value(reader);
        file.metadata.emplace(key, std::move(value));
    }

    file.tensors.reserve(static_cast<size_t>(file.n_tensors));
    for (uint64_t i = 0; i < file.n_tensors; ++i) {
        GGUFTensor tensor;
        tensor.name = reader.read_string();
        const uint32_t n_dims = reader.read<uint32_t>();
        tensor.shape.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            tensor.shape[d] = reader.read<uint64_t>();
        }
        tensor.type = static_cast<GGMLType>(reader.read<uint32_t>());
        tensor.offset = reader.read<uint64_t>();
        tensor.n_bytes = tensor_n_bytes(tensor.type, tensor.shape);
        tensor.data = nullptr;
        file.tensors.push_back(std::move(tensor));
    }

    size_t data_offset = reader.position();
    const size_t align = 32;
    if (data_offset % align != 0) {
        data_offset += align - (data_offset % align);
    }
    if (data_offset > file.mmap_size_) {
        throw std::runtime_error(
            "GGUF: data section alignment past end of file " + path.string() +
            " at offset " + std::to_string(data_offset));
    }

    const auto* data_base = bytes + data_offset;
    for (auto& tensor : file.tensors) {
        const size_t end = static_cast<size_t>(tensor.offset) + tensor.n_bytes;
        if (end > file.mmap_size_ - data_offset) {
            throw std::runtime_error(
                "GGUF: tensor '" + tensor.name + "' data extends past file end (offset " +
                std::to_string(tensor.offset) + ", nbytes " + std::to_string(tensor.n_bytes) +
                ") in file " + path.string());
        }
        tensor.data = data_base + tensor.offset;
    }

#if !defined(_WIN32) && defined(MADV_RANDOM)
    madvise(file.mmap_base_, file.mmap_size_, MADV_RANDOM);
#endif

    return file;
}

}  // namespace llm
