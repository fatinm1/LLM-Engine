#include "core/kv_cache.h"

#include <stdexcept>

namespace llm {

void KVCache::init(size_t max_seq_len, size_t n_layers, size_t n_kv_heads, size_t head_dim)
{
    max_seq_len_ = max_seq_len;
    n_layers_ = n_layers;
    n_kv_heads_ = n_kv_heads;
    head_dim_ = head_dim;
    pos_stride_ = n_kv_heads * head_dim;
    layer_stride_ = max_seq_len * pos_stride_;
    const size_t total = n_layers * layer_stride_;
    keys_.assign(total, 0.0f);
    vals_.assign(total, 0.0f);
    current_pos_ = 0;
}

float* KVCache::key_at(size_t layer, size_t pos)
{
    if (layer >= n_layers_ || pos >= max_seq_len_) {
        throw std::out_of_range("KVCache::key_at layer=" + std::to_string(layer) +
                                " pos=" + std::to_string(pos));
    }
    return keys_.data() + layer * layer_stride_ + pos * pos_stride_;
}

float* KVCache::val_at(size_t layer, size_t pos)
{
    if (layer >= n_layers_ || pos >= max_seq_len_) {
        throw std::out_of_range("KVCache::val_at layer=" + std::to_string(layer) +
                                " pos=" + std::to_string(pos));
    }
    return vals_.data() + layer * layer_stride_ + pos * pos_stride_;
}

const float* KVCache::key_at(size_t layer, size_t pos) const
{
    return const_cast<KVCache*>(this)->key_at(layer, pos);
}

const float* KVCache::val_at(size_t layer, size_t pos) const
{
    return const_cast<KVCache*>(this)->val_at(layer, pos);
}

void KVCache::advance()
{
    if (current_pos_ < max_seq_len_) {
        ++current_pos_;
    }
}

void KVCache::clear()
{
    current_pos_ = 0;
}

size_t KVCache::size() const
{
    return current_pos_;
}

bool KVCache::is_full() const
{
    return current_pos_ >= max_seq_len_;
}

}  // namespace llm
