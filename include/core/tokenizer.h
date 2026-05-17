#pragma once

#include "core/gguf_parser.h"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace llm {

using TokenID = int32_t;
constexpr TokenID TOKEN_BOS = 1;
constexpr TokenID TOKEN_EOS = 2;

class Tokenizer {
public:
    explicit Tokenizer(const GGUFFile& gguf);
    std::vector<TokenID> encode(const std::string& text, bool add_bos = true) const;
    std::string          decode(const std::vector<TokenID>& ids) const;
    std::string          decode_token(TokenID id) const;
    TokenID bos_id() const;
    TokenID eos_id() const;
    bool    is_eos(TokenID id) const;
    size_t  vocab_size() const;

private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, TokenID> token_to_id_;
    std::unordered_map<std::string, float> merge_scores_;
    std::array<TokenID, 256> byte_tokens_{};
    TokenID bos_id_ = TOKEN_BOS;
    TokenID eos_id_ = TOKEN_EOS;

    static std::vector<std::string> utf8_chars(const std::string& text);
    static std::string normalize_text(const std::string& text);
};

}  // namespace llm
