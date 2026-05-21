#include "ui/app.h"
#include "imgui.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace llm::ui {

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
    const float messages_h = ImGui::GetContentRegionAvail().y - input_h - 12.f;

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

    state_ = AppState::GENERATING;
    stop_flag_.store(false);
    tps_.store(0.f);
    tokens_generated_.store(0);

    {
        std::lock_guard<std::mutex> lk(msg_mutex_);
        messages_.push_back({Message::Role::User, prompt, true});
        messages_.push_back({Message::Role::Assistant, "", false});
        scroll_to_bottom_ = true;
    }

    if (gen_thread_.joinable()) {
        gen_thread_.join();
    }

    gen_thread_ = std::thread([this, prompt]() {
        try {
            model_->generate(
                prompt, sampler_cfg_, 512,
                [this](llm::TokenID, const std::string& piece) -> bool {
                    {
                        std::lock_guard<std::mutex> lk(msg_mutex_);
                        messages_.back().text += piece;
                        scroll_to_bottom_ = true;
                    }
                    tps_.store(model_->tokens_per_second());
                    tokens_generated_.store(model_->tokens_generated());
                    return !stop_flag_.load();
                });
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lk(msg_mutex_);
            messages_.back().text += "\n[Error: ";
            messages_.back().text += e.what();
            messages_.back().text += "]";
        }
        {
            std::lock_guard<std::mutex> lk(msg_mutex_);
            if (!messages_.empty()) {
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
