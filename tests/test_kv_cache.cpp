#include "core/kv_cache.h"

#include <gtest/gtest.h>

using namespace llm;

TEST(KVCache, InitEmpty)
{
    KVCache cache;
    cache.init(4, 2, 2, 3);
    EXPECT_EQ(cache.size(), 0u);
    EXPECT_FALSE(cache.is_full());
}

TEST(KVCache, WriteReadPattern)
{
    KVCache cache;
    cache.init(8, 2, 2, 4);
    float* k0 = cache.key_at(0, 0);
    for (int i = 0; i < 8; ++i) {
        k0[i] = static_cast<float>(i);
    }
    const float* k0r = cache.key_at(0, 0);
    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(k0r[i], static_cast<float>(i));
    }
}

TEST(KVCache, LayersDoNotAlias)
{
    KVCache cache;
    cache.init(8, 2, 2, 4);
    cache.key_at(0, 0)[0] = 1.0f;
    cache.key_at(1, 0)[0] = 2.0f;
    EXPECT_FLOAT_EQ(cache.key_at(0, 0)[0], 1.0f);
    EXPECT_FLOAT_EQ(cache.key_at(1, 0)[0], 2.0f);
}

TEST(KVCache, AdvanceUntilFull)
{
    KVCache cache;
    cache.init(3, 1, 1, 2);
    cache.advance();
    cache.advance();
    EXPECT_FALSE(cache.is_full());
    cache.advance();
    EXPECT_TRUE(cache.is_full());
    EXPECT_EQ(cache.size(), 3u);
}

TEST(KVCache, ClearResetsSize)
{
    KVCache cache;
    cache.init(4, 1, 1, 2);
    cache.advance();
    cache.advance();
    cache.clear();
    EXPECT_EQ(cache.size(), 0u);
}
