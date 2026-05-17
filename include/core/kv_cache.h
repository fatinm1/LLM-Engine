#pragma once

#include <cstddef>
#include <vector>

namespace llm {

class KVCache {
public:
    void init(size_t max_seq_len, size_t n_layers, size_t n_kv_heads, size_t head_dim);
    float*       key_at(size_t layer, size_t pos);
    float*       val_at(size_t layer, size_t pos);
    const float* key_at(size_t layer, size_t pos) const;
    const float* val_at(size_t layer, size_t pos) const;
    void   advance();
    void   clear();
    size_t size() const;
    bool   is_full() const;

private:
    std::vector<float> keys_;
    std::vector<float> vals_;
    size_t max_seq_len_ = 0;
    size_t n_layers_ = 0;
    size_t n_kv_heads_ = 0;
    size_t head_dim_ = 0;
    size_t pos_stride_ = 0;
    size_t layer_stride_ = 0;
    size_t current_pos_ = 0;
};

}  // namespace llm
