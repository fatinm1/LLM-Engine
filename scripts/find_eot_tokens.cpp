#include "core/gguf_parser.h"
#include "core/model.h"
#include "core/tokenizer.h"
#include <iostream>

int main()
{
    auto gguf = llm::GGUFParser::parse("models/Llama-3.2-1B-Instruct-Q4_K_M.gguf");
    llm::Tokenizer tok(gguf);

    const std::string prompt = llm::format_llama3_prompt("What is the capital of France?");
    const auto ids = tok.encode(prompt, true);
    std::cout << "eos_id=" << tok.eos_id() << " encoded=" << ids.size() << "\n";
    for (const char* needle : {"<|", "eot", "end_of", "assistant", "header"}) {
        for (size_t i = 0; i < tok.vocab_size(); ++i) {
            const std::string s = tok.decode_token(static_cast<llm::TokenID>(i));
            if (s.find(needle) != std::string::npos && s.size() < 32) {
                std::cout << i << " '" << s << "' eos=" << tok.is_eos(static_cast<llm::TokenID>(i))
                          << "\n";
            }
        }
    }
    return 0;
}
