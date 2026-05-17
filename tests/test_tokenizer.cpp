#include "core/gguf_parser.h"
#include "core/tokenizer.h"

#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>

using namespace llm;

static std::filesystem::path model_path()
{
    if (const char* env = std::getenv("LLM_MODEL_PATH")) {
        return env;
    }
    return "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
}

TEST(Tokenizer, WithRealModel)
{
    const auto path = model_path();
    if (!std::filesystem::exists(path)) {
        GTEST_SKIP() << "Model not found at " << path;
    }

    GGUFFile gguf = GGUFParser::parse(path);
    Tokenizer tok(gguf);

    const auto ids = tok.encode("Hello, world!", true);
    ASSERT_FALSE(ids.empty());
    EXPECT_EQ(ids.front(), tok.bos_id());

    const std::string roundtrip = tok.decode(ids);
    EXPECT_EQ(roundtrip, "Hello, world!");

    const auto empty_ids = tok.encode("", true);
    ASSERT_EQ(empty_ids.size(), 1u);
    EXPECT_EQ(empty_ids.front(), tok.bos_id());

    EXPECT_TRUE(tok.is_eos(tok.eos_id()));
}
