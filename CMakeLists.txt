cmake_minimum_required(VERSION 3.20)
project(llm_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/arch:AVX2 /O2)
else()
    add_compile_options(-O3 -ffast-math)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|x86_64|i[3-6]86)$")
        add_compile_options(-march=native -mavx2 -mfma)
    elseif(NOT APPLE)
        add_compile_options(-march=native)
    endif()
endif()

set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/third_party/imgui)
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/glfw)

add_library(llm_core STATIC
    src/core/simd_math.cpp
    src/core/gguf_parser.cpp
    src/core/tokenizer.cpp
    src/core/kv_cache.cpp
    src/core/sampler.cpp
    src/core/model.cpp
)
target_include_directories(llm_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

find_package(OpenGL REQUIRED)

add_executable(llm_desktop
    src/ui/main.cpp
    ${IMGUI_SOURCES}
)
target_include_directories(llm_desktop PRIVATE
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(llm_desktop PRIVATE
    llm_core
    glfw
    OpenGL::GL
)
if(APPLE)
    target_link_libraries(llm_desktop PRIVATE
        "-framework IOKit"
        "-framework Cocoa"
        "-framework CoreVideo"
    )
endif()

enable_testing()
add_subdirectory(tests)
