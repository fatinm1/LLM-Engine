include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(llm_tests
    test_simd_math.cpp
    test_kv_cache.cpp
    test_sampler.cpp
    test_tokenizer.cpp
)
target_link_libraries(llm_tests PRIVATE GTest::gtest_main llm_core)
target_include_directories(llm_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
include(GoogleTest)
gtest_discover_tests(llm_tests)
