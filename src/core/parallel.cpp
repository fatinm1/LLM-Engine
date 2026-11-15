#include "core/parallel.h"

#include <algorithm>
#include <chrono>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif

namespace llm {

namespace {

inline void cpu_relax_fwd()
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#endif
}

// How long an idle thread spins before sleeping. Time-based because a spin
// iteration costs ~1 cycle on ARM (`yield`) but ~100 cycles on x86 (`pause`).
constexpr auto kSpinTime = std::chrono::microseconds(500);

// Spin until pred() holds or kSpinTime passes. Returns pred().
template <typename Pred>
bool spin_until(Pred pred)
{
    const auto deadline = std::chrono::steady_clock::now() + kSpinTime;
    for (;;) {
        for (int i = 0; i < 256; ++i) {
            if (pred()) {
                return true;
            }
            cpu_relax_fwd();
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return pred();
        }
    }
}


}  // namespace

WorkerPool::WorkerPool(size_t n_threads)
{
    if (n_threads == 0) {
        n_threads = std::max<size_t>(1, std::thread::hardware_concurrency());
    }
    workers_.reserve(n_threads - 1);
    for (size_t i = 0; i + 1 < n_threads; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

WorkerPool::~WorkerPool()
{
    {
        std::lock_guard<std::mutex> lk(mtx_);
        stop_.store(true, std::memory_order_release);
    }
    cv_start_.notify_all();
    for (auto& w : workers_) {
        w.join();
    }
}

void WorkerPool::run_chunks()
{
    for (;;) {
        const size_t begin = next_.fetch_add(grain_, std::memory_order_relaxed);
        if (begin >= n_) {
            return;
        }
        (*fn_)(begin, std::min(begin + grain_, n_));
    }
}

void WorkerPool::worker_loop()
{
    uint64_t seen = 0;
    for (;;) {
        auto ready = [&] {
            return generation_.load(std::memory_order_acquire) != seen ||
                   stop_.load(std::memory_order_acquire);
        };
        if (!spin_until(ready)) {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_start_.wait(lk, ready);
        }
        if (stop_.load(std::memory_order_acquire)) {
            return;
        }
        seen = generation_.load(std::memory_order_acquire);

        run_chunks();

        if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lk(mtx_);
            cv_done_.notify_one();
        }
    }
}

void WorkerPool::parallel_for(size_t n, size_t grain, const RangeFn& fn)
{
    if (n == 0) {
        return;
    }
    grain = std::max<size_t>(1, grain);
    if (workers_.empty() || n <= grain) {
        fn(0, n);
        return;
    }

    fn_ = &fn;
    n_ = n;
    grain_ = grain;
    next_.store(0, std::memory_order_relaxed);
    pending_.store(workers_.size(), std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(mtx_);
        generation_.fetch_add(1, std::memory_order_release);
    }
    cv_start_.notify_all();

    run_chunks();

    auto done = [&] { return pending_.load(std::memory_order_acquire) == 0; };
    if (!spin_until(done)) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_done_.wait(lk, done);
    }
    fn_ = nullptr;
}

}  // namespace llm
