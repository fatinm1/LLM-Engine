#include "ui/app.h"
#include "imgui.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace llm::ui {

static std::string pick_file_macos()
{
    FILE* pipe = popen(
        "osascript -e 'set f to choose file with prompt \"Select a text file\"' "
        "-e 'POSIX path of f' 2>/dev/null",
        "r");
    if (!pipe) {
        return "";
    }
    char buf[4096] = {};
    if (!fgets(buf, sizeof(buf), pipe)) {
        pclose(pipe);
        return "";
    }
    pclose(pipe);
    std::string path(buf);
    while (!path.empty() && (path.back() == '\n' || path.back() == '\r')) {
        path.pop_back();
    }
    return path;
}

static std::string read_file(const std::string& path, size_t max_bytes = 4096)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        return "";
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (content.size() > max_bytes) {
        content = content.substr(0, max_bytes) + "\n... [truncated at 4KB]";
    }
    return content;
}

App::App()
{
    sampler_cfg_.temperature = 0.8f;
    sampler_cfg_.top_k = 40;
    sampler_cfg_.top_p = 0.95f;
    sampler_cfg_.repeat_penalty = 1.1f;
    std::memset(path_buf_, 0, sizeof(path_buf_));
    std::memset(input_buf_, 0, sizeof(input_buf_));
}

App::~App()
{
    stop_generation();
    if (gen_thread_.joinable()) {
        gen_thread_.join();
    }
}

void App::render_frame()
{
    render_menu_bar();

    ImGuiIO& io = ImGui::GetIO();
    const float sidebar_w = 280.f;
    const float status_h = 24.f;
    const float content_h = io.DisplaySize.y - ImGui::GetFrameHeight() - status_h;

    ImGui::SetNextWindowPos({0.f, ImGui::GetFrameHeight()});
    ImGui::SetNextWindowSize({sidebar_w, content_h});
    ImGui::Begin("##sidebar", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus);
    render_sidebar();
    ImGui::End();

    ImGui::SetNextWindowPos({sidebar_w, ImGui::GetFrameHeight()});
    ImGui::SetNextWindowSize({io.DisplaySize.x - sidebar_w, content_h});
    ImGui::Begin("##chat", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus);
    render_chat();
    ImGui::End();

    ImGui::SetNextWindowPos({0.f, io.DisplaySize.y - status_h});
    ImGui::SetNextWindowSize({io.DisplaySize.x, status_h});
    ImGui::Begin("##status", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);
    render_status_bar();
    ImGui::End();
}

void App::render_menu_bar()
{
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("New Conversation", "Ctrl+N", false, state_ == AppState::READY)) {
                std::lock_guard<std::mutex> lk(msg_mutex_);
                messages_.clear();
                if (model_) {
                    model_->reset();
                }
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Quit", "Alt+F4")) {
                quit_ = true;
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

void App::render_sidebar()
{
    ImGui::TextDisabled("MODEL");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##path", path_buf_, sizeof(path_buf_));
    ImGui::SameLine();

    const bool loading = (state_ == AppState::LOADING);
    if (loading) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button("Load", {-1, 0})) {
        if (path_buf_[0] != '\0') {
            load_model(path_buf_);
        }
    }
    if (loading) {
        ImGui::EndDisabled();
    }

    ImGui::Spacing();

    if (state_ == AppState::READY || state_ == AppState::GENERATING) {
        if (model_) {
            const auto& cfg = model_->config();
            ImGui::Text("Layers:  %u", cfg.n_layers);
            ImGui::Text("Heads:   %u / %u (GQA)", cfg.n_heads, cfg.n_kv_heads);
            ImGui::Text("Dim:     %u", cfg.embed_dim);
            ImGui::Text("Vocab:   %u", cfg.vocab_size);
            ImGui::Text("Context: %u", cfg.max_seq_len);
        }
    } else if (state_ == AppState::LOADING) {
        ImGui::TextColored({1.f, 0.8f, 0.f, 1.f}, "Loading...");
    } else if (state_ == AppState::ERROR) {
        ImGui::TextColored({1.f, 0.3f, 0.3f, 1.f}, "Error:");
        ImGui::TextWrapped("%s", error_msg_.c_str());
    } else {
        ImGui::TextDisabled("No model loaded.");
        ImGui::TextWrapped("Enter a path to a .gguf file above and click Load.");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextDisabled("GENERATION");
    ImGui::Spacing();

    float temp = sampler_cfg_.temperature;
    if (ImGui::SliderFloat("Temperature", &temp, 0.0f, 2.0f, "%.2f")) {
        sampler_cfg_.temperature = temp;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("0 = deterministic\nHigher = more creative");
    }

    int top_k = sampler_cfg_.top_k;
    if (ImGui::SliderInt("Top-K", &top_k, 0, 100)) {
        sampler_cfg_.top_k = top_k;
    }

    float top_p = sampler_cfg_.top_p;
    if (ImGui::SliderFloat("Top-P", &top_p, 0.0f, 1.0f, "%.2f")) {
        sampler_cfg_.top_p = top_p;
    }

    float rep = sampler_cfg_.repeat_penalty;
    if (ImGui::SliderFloat("Repeat Penalty", &rep, 1.0f, 2.0f, "%.2f")) {
        sampler_cfg_.repeat_penalty = rep;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextDisabled("STATS");
    ImGui::Spacing();
    ImGui::Text("Speed:  %.1f tok/s", tps_.load());
    ImGui::Text("Tokens: %zu", tokens_generated_.load());
}

void App::render_chat()
{
    const float input_h = 80.f;
    const float btn_w = 80.f;
    const float attach_h = 32.f;
    const float messages_h = ImGui::GetContentRegionAvail().y - input_h - attach_h - 12.f;

    ImGui::BeginChild("##messages", {0.f, messages_h}, false);
    {
        std::lock_guard<std::mutex> lk(msg_mutex_);
        for (const auto& msg : messages_) {
            if (msg.role == Message::Role::User) {
                ImGui::TextDisabled("You");
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.9f, 1.f));
            } else {
                ImGui::TextDisabled("Assistant");
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.9f, 0.6f, 1.f));
            }
            ImGui::TextWrapped("%s", msg.text.c_str());
            ImGui::PopStyleColor();
            if (!msg.complete && msg.role == Message::Role::Assistant) {
                ImGui::TextDisabled("...");
            }
            ImGui::Spacing();
        }
        if (scroll_to_bottom_) {
            ImGui::SetScrollHereY(1.0f);
            scroll_to_bottom_ = false;
        }
    }
    ImGui::EndChild();

    ImGui::Separator();

    if (ImGui::Button("+ Attach file")) {
        const std::string path = pick_file_macos();
        if (!path.empty()) {
            const std::string content = read_file(path);
            if (!content.empty()) {
                const size_t slash = path.rfind('/');
                attached_file_name_ = (slash != std::string::npos) ? path.substr(slash + 1) : path;
                attached_file_content_ = content;
            }
        }
    }

    if (!attached_file_name_.empty()) {
        ImGui::SameLine();
        ImGui::TextColored({0.4f, 0.8f, 1.0f, 1.0f}, "[%s]", attached_file_name_.c_str());
        ImGui::SameLine();
        if (ImGui::SmallButton("x")) {
            attached_file_name_.clear();
            attached_file_content_.clear();
        }
    }

    ImGui::Spacing();

    bool send = false;
    const float input_w = ImGui::GetContentRegionAvail().x - btn_w - 8.f;
    ImGui::SetNextItemWidth(input_w);
    if (ImGui::InputTextMultiline("##input", input_buf_, sizeof(input_buf_), {input_w, input_h},
                                  ImGuiInputTextFlags_EnterReturnsTrue)) {
        send = true;
    }

    ImGui::SameLine();

    if (state_ == AppState::GENERATING) {
        if (ImGui::Button("Stop", {btn_w, input_h})) {
            stop_generation();
        }
    } else {
        const bool can_send = (state_ == AppState::READY) && (input_buf_[0] != '\0');
        if (!can_send) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Send", {btn_w, input_h})) {
            send = true;
        }
        if (!can_send) {
            ImGui::EndDisabled();
        }
    }

    if (send && state_ == AppState::READY && input_buf_[0] != '\0') {
        submit_prompt(std::string(input_buf_));
        std::memset(input_buf_, 0, sizeof(input_buf_));
    }
}

void App::render_status_bar()
{
    switch (state_) {
    case AppState::NO_MODEL:
        ImGui::TextDisabled("No model loaded");
        break;
    case AppState::LOADING:
        ImGui::TextColored({1.f, 0.8f, 0.f, 1.f}, "Loading model...");
        break;
    case AppState::READY:
        ImGui::TextColored({0.4f, 0.9f, 0.4f, 1.f}, "Ready");
        break;
    case AppState::GENERATING:
        ImGui::TextColored({0.9f, 0.8f, 0.2f, 1.f}, "Generating...  %.1f tok/s", tps_.load());
        break;
    case AppState::ERROR:
        ImGui::TextColored({1.f, 0.3f, 0.3f, 1.f}, "Error");
        break;
    }
}

void App::auto_load(const std::string& path)
{
    std::strncpy(path_buf_, path.c_str(), sizeof(path_buf_) - 1);
    path_buf_[sizeof(path_buf_) - 1] = '\0';
    load_model(path);
}

void App::load_model(const std::string& path)
{
    if (state_ == AppState::LOADING || state_ == AppState::GENERATING) {
        return;
    }

    state_ = AppState::LOADING;
    error_msg_.clear();

    std::thread([this, path]() {
        try {
            auto gguf = std::make_unique<GGUFFile>(GGUFParser::parse(path));
            auto model = llm::Model::load(*gguf);
            gguf_ = std::move(gguf);
            model_ = std::move(model);
            state_ = AppState::READY;
        } catch (const std::exception& e) {
            error_msg_ = e.what();
            state_ = AppState::ERROR;
        }
    }).detach();
}

void App::submit_prompt(const std::string& prompt)
{
    if (!model_ || state_ != AppState::READY) {
        return;
    }

    std::string full_prompt = prompt;
    std::string display_prompt = prompt;

    if (!attached_file_content_.empty()) {
        full_prompt = "I'm sharing a file called '" + attached_file_name_ +
                      "'. Here are its contents:\n\n```\n" + attached_file_content_ +
                      "\n```\n\n" + prompt;
        display_prompt = "[" + attached_file_name_ + "] " + prompt;
        attached_file_name_.clear();
        attached_file_content_.clear();
    }

    const std::string formatted = llm::format_llama3_prompt(full_prompt);

    state_ = AppState::GENERATING;
    stop_flag_.store(false);
    tps_.store(0.f);
    tokens_generated_.store(0);

    {
        std::lock_guard<std::mutex> lk(msg_mutex_);
        messages_.push_back({Message::Role::User, display_prompt, true});
        messages_.push_back({Message::Role::Assistant, "", false});
        scroll_to_bottom_ = true;
    }

    if (gen_thread_.joinable()) {
        gen_thread_.join();
    }

    gen_thread_ = std::thread([this, formatted]() {
        try {
            model_->generate(
                formatted, sampler_cfg_, 512,
                [this](llm::TokenID, const std::string& piece) -> bool {
                    {
                        std::lock_guard<std::mutex> lk(msg_mutex_);
                        messages_.back().text += piece;
                        scroll_to_bottom_ = true;
                    }
                    tps_.store(model_->tokens_per_second());
                    tokens_generated_.store(model_->tokens_generated());
                    return !stop_flag_.load();
                },
                [this]() -> bool { return stop_flag_.load(); });
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lk(msg_mutex_);
            messages_.back().text += "\n[Error: ";
            messages_.back().text += e.what();
            messages_.back().text += "]";
        }
        {
            std::lock_guard<std::mutex> lk(msg_mutex_);
            if (!messages_.empty()) {
                if (stop_flag_.load() && messages_.back().text.empty()) {
                    messages_.back().text = "[Stopped]";
                }
                messages_.back().complete = true;
            }
        }
        state_ = AppState::READY;
    });
}

void App::stop_generation()
{
    stop_flag_.store(true);
}

} // namespace llm::ui
