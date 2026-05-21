#pragma once
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "core/model.h"
#include "core/sampler.h"
#include "core/gguf_parser.h"

namespace llm::ui {

enum class AppState {
    NO_MODEL,
    LOADING,
    READY,
    GENERATING,
    ERROR
};

struct Message {
    enum class Role { User, Assistant };
    Role        role;
    std::string text;
    bool        complete = false;
};

class App {
public:
    App();
    ~App();

    // Call every frame from the render loop
    void render_frame();

    bool should_quit() const { return quit_; }

private:
    void render_menu_bar();
    void render_sidebar();
    void render_chat();
    void render_status_bar();

    void load_model(const std::string& path);
    void submit_prompt(const std::string& prompt);
    void stop_generation();

    // State
    AppState                     state_     = AppState::NO_MODEL;
    std::unique_ptr<GGUFFile>    gguf_;
    std::unique_ptr<llm::Model>  model_;

    // Chat
    std::vector<Message>         messages_;
    std::mutex                   msg_mutex_;
    bool                         scroll_to_bottom_ = false;

    // Input
    char                         input_buf_[4096] = {};
    bool                         quit_            = false;

    // Model path input (simple text field — no native file dialog)
    char                         path_buf_[1024]  = {};

    // Sampler config (controlled by UI sliders)
    llm::SamplerConfig           sampler_cfg_;

    // Generation thread
    std::thread                  gen_thread_;
    std::atomic<bool>            stop_flag_{ false };

    // Stats (updated from gen thread, read by UI thread)
    std::atomic<float>           tps_{ 0.f };
    std::atomic<size_t>          tokens_generated_{ 0 };

    // Error message
    std::string                  error_msg_;
};

} // namespace llm::ui
