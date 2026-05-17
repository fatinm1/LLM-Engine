#include "core/tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace llm {

namespace {

const char* kSpaceToken = "\xe2\x96\x81";

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

std::vector<float> gguf_float_array(const GGUFFile& gguf, const std::string& key)
{
    const auto it = gguf.metadata.find(key);
    if (it == gguf.metadata.end()) {
        throw std::runtime_error("GGUF metadata missing array key: " + key);
    }
    const auto* arr = std::get_if<GGUFArray>(&it->second);
    if (arr == nullptr) {
        throw std::runtime_error("GGUF metadata key is not an array: " + key);
    }
    std::vector<float> out;
    out.reserve(arr->values.size());
    for (const auto& v : arr->values) {
        if (const auto* f = std::get_if<float>(&v)) {
            out.push_back(*f);
        } else if (const auto* d = std::get_if<double>(&v)) {
            out.push_back(static_cast<float>(*d));
        } else {
            throw std::runtime_error("GGUF array element for " + key + " is not float");
        }
    }
    return out;
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

Tokenizer::Tokenizer(const GGUFFile& gguf)
{
    vocab_ = gguf_string_array(gguf, "tokenizer.ggml.tokens");
    const std::vector<float> scores = gguf_float_array(gguf, "tokenizer.ggml.scores");

    if (scores.size() != vocab_.size()) {
        throw std::runtime_error(
            "tokenizer.ggml.scores size " + std::to_string(scores.size()) +
            " != vocab size " + std::to_string(vocab_.size()));
    }

    for (size_t i = 0; i < vocab_.size(); ++i) {
        token_to_id_[vocab_[i]] = static_cast<TokenID>(i);
    }

    byte_tokens_.fill(-1);
    for (size_t b = 0; b < 256; ++b) {
        char label[8];
        std::snprintf(label, sizeof(label), "<0x%02X>", static_cast<int>(b));
        const auto it = token_to_id_.find(label);
        if (it != token_to_id_.end()) {
            byte_tokens_[b] = it->second;
        }
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

std::string Tokenizer::normalize_text(const std::string& text)
{
    std::string out;
    out.reserve(text.size() + 8);
    bool at_word_start = true;
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        const bool is_space = (c == ' ' || c == '\t' || c == '\n' || c == '\r');
        if (is_space) {
            at_word_start = true;
            ++i;
            continue;
        }
        if (at_word_start) {
            out += kSpaceToken;
            at_word_start = false;
        }
        if ((c & 0x80) == 0) {
            out.push_back(static_cast<char>(c));
            ++i;
        } else {
            size_t len = 1;
            if ((c & 0xE0) == 0xC0) {
                len = 2;
            } else if ((c & 0xF0) == 0xE0) {
                len = 3;
            } else if ((c & 0xF8) == 0xF0) {
                len = 4;
            }
            out.append(text.substr(i, len));
            i += len;
        }
    }
    return out;
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
    std::vector<TokenID> ids;
    if (add_bos) {
        ids.push_back(bos_id_);
    }
    if (text.empty()) {
        return ids;
    }

    const std::string normalized = normalize_text(text);
    std::vector<std::string> symbols;
    for (const std::string& ch : utf8_chars(normalized)) {
        const auto it = token_to_id_.find(ch);
        if (it != token_to_id_.end()) {
            symbols.push_back(ch);
        } else {
            for (unsigned char b : ch) {
                const TokenID bt = byte_tokens_[b];
                if (bt >= 0) {
                    symbols.push_back(vocab_[static_cast<size_t>(bt)]);
                } else {
                    throw std::runtime_error("No byte token for value " + std::to_string(b));
                }
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
            const auto tok_it = token_to_id_.find(candidate);
            if (tok_it == token_to_id_.end()) {
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
        ids.push_back(it->second);
    }
    return ids;
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
        std::string piece = decode_token(id);
        size_t pos = 0;
        while ((pos = piece.find(kSpaceToken, pos)) != std::string::npos) {
            piece.replace(pos, 3, " ");
            pos += 1;
        }
        out += piece;
    }
    if (!out.empty() && out[0] == ' ') {
        out.erase(0, 1);
    }
    return out;
}

}  // namespace llm
