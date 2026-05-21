#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

namespace llm {

class ThreadPool {
public:
    explicit ThreadPool(size_t n_threads = 0);
    ~ThreadPool();

    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>;

    void wait_all();
    size_t n_threads() const;

private:
    void worker_loop();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::condition_variable cv_done_;
    size_t active_ = 0;
    bool stop_ = false;
};

template<typename F, typename... Args>
auto ThreadPool::submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>
{
    using ReturnType = std::invoke_result_t<F, Args...>;
    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<ReturnType> result = task->get_future();
    {
        std::lock_guard<std::mutex> lk(mtx_);
        ++active_;
        tasks_.emplace([task]() { (*task)(); });
    }
    cv_.notify_one();
    return result;
}

} // namespace llm
