#include "core/thread_pool.h"

#include <algorithm>
#include <stdexcept>

namespace llm {

ThreadPool::ThreadPool(size_t n_threads)
{
    if (n_threads == 0) {
        n_threads = std::min(
            static_cast<size_t>(std::thread::hardware_concurrency()),
            static_cast<size_t>(8));
    }
    workers_.reserve(n_threads);
    for (size_t i = 0; i < n_threads; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

ThreadPool::~ThreadPool()
{
    {
        std::lock_guard<std::mutex> lk(mtx_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) {
        w.join();
    }
}

void ThreadPool::worker_loop()
{
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) {
                return;
            }
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
        {
            std::lock_guard<std::mutex> lk(mtx_);
            --active_;
        }
        cv_done_.notify_all();
    }
}

void ThreadPool::wait_all()
{
    std::unique_lock<std::mutex> lk(mtx_);
    cv_done_.wait(lk, [this] { return active_ == 0 && tasks_.empty(); });
}

size_t ThreadPool::n_threads() const
{
    return workers_.size();
}

} // namespace llm
