#include "core/tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace llm {

namespace {

std::vector<std::string> gguf_string_array(const GGUFFile& gguf, const std::string& key)
{
    const auto it = gguf.metadata.find(key);
    if (it == gguf.metadata.end()) {
        throw std::runtime_error("GGUF metadata missing array key: " + key);
    }
    const auto* arr = std::get_if<GGUFArray>(&it->second);
    if (arr == nullptr) {
        throw std::runtime_error("GGUF metadata key is not an array: " + key);
    }
    std::vector<std::string> out;
    out.reserve(arr->values.size());
    for (const auto& v : arr->values) {
        if (const auto* s = std::get_if<std::string>(&v)) {
            out.push_back(*s);
        } else {
            throw std::runtime_error("GGUF array element for " + key + " is not a string");
        }
    }
    return out;
}

constexpr TokenID TOKEN_UNK = 0;

enum class TokenType : int32_t {
    NORMAL = 1,
    BYTE = 6,
};

std::vector<float> load_optional_scores(const GGUFFile& gguf, size_t vocab_size)
{
    std::vector<float> scores(vocab_size, 0.0f);
    const auto it = gguf.metadata.find("tokenizer.ggml.scores");
    if (it == gguf.metadata.end()) {
        return scores;
    }
    const auto* arr = std::get_if<GGUFArray>(&it->second);
    if (arr == nullptr) {
        return scores;
    }
    const size_t n = std::min(vocab_size, arr->values.size());
    for (size_t i = 0; i < n; ++i) {
        const auto& v = arr->values[i];
        if (const auto* f = std::get_if<float>(&v)) {
            scores[i] = *f;
        } else if (const auto* d = std::get_if<double>(&v)) {
            scores[i] = static_cast<float>(*d);
        }
    }
    return scores;
}

std::vector<int32_t> load_optional_token_types(const GGUFFile& gguf, size_t vocab_size)
{
    std::vector<int32_t> types(vocab_size, static_cast<int32_t>(TokenType::NORMAL));
    const auto it = gguf.metadata.find("tokenizer.ggml.token_type");
    if (it == gguf.metadata.end()) {
        return types;
    }
    const auto* arr = std::get_if<GGUFArray>(&it->second);
    if (arr == nullptr) {
        return types;
    }
    const size_t n = std::min(vocab_size, arr->values.size());
    for (size_t i = 0; i < n; ++i) {
        const auto& v = arr->values[i];
        if (const auto* t = std::get_if<int32_t>(&v)) {
            types[i] = *t;
        } else if (const auto* t = std::get_if<uint32_t>(&v)) {
            types[i] = static_cast<int32_t>(*t);
        } else if (const auto* t = std::get_if<int16_t>(&v)) {
            types[i] = static_cast<int32_t>(*t);
        } else if (const auto* t = std::get_if<uint16_t>(&v)) {
            types[i] = static_cast<int32_t>(*t);
        }
    }
    return types;
}

uint32_t metadata_token_id(const GGUFFile& gguf, const std::string& key, uint32_t def)
{
    const auto it = gguf.metadata.find(key);
    if (it == gguf.metadata.end()) {
        return def;
    }
    if (const auto* scalar = std::get_if<GGUFScalar>(&it->second)) {
        if (const auto* v = std::get_if<uint32_t>(scalar)) {
            return *v;
        }
        if (const auto* v = std::get_if<int32_t>(scalar)) {
            return static_cast<uint32_t>(*v);
        }
    }
    return def;
}

}  // namespace

std::string Tokenizer::normalize_text(const std::string& text)
{
    return text;
}

Tokenizer::Tokenizer(const GGUFFile& gguf)
{
    vocab_ = gguf_string_array(gguf, "tokenizer.ggml.tokens");
    const std::vector<float> scores = load_optional_scores(gguf, vocab_.size());
    const std::vector<int32_t> token_types = load_optional_token_types(gguf, vocab_.size());

    for (size_t i = 0; i < vocab_.size(); ++i) {
        token_to_id_[vocab_[i]] = static_cast<TokenID>(i);
    }

    byte_tokens_.fill(TOKEN_UNK);
    for (size_t b = 0; b < 256; ++b) {
        char label[8];
        std::snprintf(label, sizeof(label), "<0x%02X>", static_cast<int>(b));
        const auto it = token_to_id_.find(label);
        if (it != token_to_id_.end()) {
            byte_tokens_[b] = it->second;
        }
    }
    for (size_t i = 0; i < vocab_.size(); ++i) {
        if (token_types[i] != static_cast<int32_t>(TokenType::BYTE)) {
            continue;
        }
        const std::string& tok = vocab_[i];
        if (tok.size() != 1) {
            continue;
        }
        const unsigned char b = static_cast<unsigned char>(tok[0]);
        byte_tokens_[b] = static_cast<TokenID>(i);
    }

    bos_id_ = static_cast<TokenID>(metadata_token_id(gguf, "tokenizer.ggml.bos_token_id", TOKEN_BOS));
    eos_id_ = static_cast<TokenID>(metadata_token_id(gguf, "tokenizer.ggml.eos_token_id", TOKEN_EOS));

    auto pair_key = [](const std::string& a, const std::string& b) { return a + "\x1f" + b; };
    for (size_t i = 0; i < vocab_.size(); ++i) {
        const std::string& tok = vocab_[i];
        for (size_t split = 1; split < tok.size(); ++split) {
            const std::string left = tok.substr(0, split);
            const std::string right = tok.substr(split);
            if (token_to_id_.count(left) && token_to_id_.count(right)) {
                const std::string key = pair_key(left, right);
                merge_scores_[key] = std::max(merge_scores_[key], scores[i]);
            }
        }
    }
}

std::vector<std::string> Tokenizer::utf8_chars(const std::string& text)
{
    std::vector<std::string> chars;
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        size_t len = 1;
        if ((c & 0x80) == 0) {
            len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
        } else {
            len = 1;
        }
        chars.push_back(text.substr(i, len));
        i += len;
    }
    return chars;
}

TokenID Tokenizer::bos_id() const { return bos_id_; }
TokenID Tokenizer::eos_id() const { return eos_id_; }
bool Tokenizer::is_eos(TokenID id) const { return id == eos_id_; }
size_t Tokenizer::vocab_size() const { return vocab_.size(); }

std::string Tokenizer::decode_token(TokenID id) const
{
    if (id < 0 || static_cast<size_t>(id) >= vocab_.size()) {
        return "";
    }
    return vocab_[static_cast<size_t>(id)];
}

std::vector<TokenID> Tokenizer::encode(const std::string& text, bool add_bos) const
{
    std::vector<TokenID> result;
    if (add_bos) {
        result.push_back(bos_id_);
    }
    if (text.empty()) {
        return result;
    }

    std::vector<std::string> symbols;
    for (const std::string& ch : utf8_chars(text)) {
        const auto it = token_to_id_.find(ch);
        if (it != token_to_id_.end()) {
            symbols.push_back(ch);
        } else {
            for (unsigned char b : ch) {
                const TokenID bt = byte_tokens_[b];
                symbols.push_back(vocab_[static_cast<size_t>(bt)]);
            }
        }
    }

    auto pair_key = [](const std::string& a, const std::string& b) { return a + "\x1f" + b; };

    bool merged = true;
    while (merged && symbols.size() > 1) {
        merged = false;
        float best_score = -1.0f;
        size_t best_idx = 0;
        std::string best_merged;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            const std::string candidate = symbols[i] + symbols[i + 1];
            if (token_to_id_.find(candidate) == token_to_id_.end()) {
                continue;
            }
            const auto score_it = merge_scores_.find(pair_key(symbols[i], symbols[i + 1]));
            if (score_it == merge_scores_.end()) {
                continue;
            }
            const float score = score_it->second;
            if (score > best_score) {
                best_score = score;
                best_idx = i;
                best_merged = candidate;
                merged = true;
            }
        }
        if (merged) {
            symbols[best_idx] = best_merged;
            symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_idx + 1));
        }
    }

    for (const std::string& sym : symbols) {
        const auto it = token_to_id_.find(sym);
        if (it == token_to_id_.end()) {
            throw std::runtime_error("Tokenizer: unknown symbol after BPE: " + sym);
        }
        result.push_back(it->second);
    }
    return result;
}

std::string Tokenizer::decode(const std::vector<TokenID>& ids) const
{
    std::string out;
    for (TokenID id : ids) {
        if (id == bos_id_) {
            continue;
        }
        if (id == eos_id_) {
            break;
        }
        out += decode_token(id);
    }
    return out;
}

}  // namespace llm
