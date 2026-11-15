#pragma once

// A fork-join worker pool for the hot loops of a forward pass.
//
// parallel_for() splits [0, n) into chunks of `grain` items that threads claim
// from a shared atomic counter, so fast and slow cores (Apple's performance
// and efficiency cores) each take as much work as they can finish. The calling
// thread works too. Workers spin briefly between jobs, because a forward pass
// issues one job per matrix (100+ per token) and waking a sleeping thread
// costs more than many of those jobs take; after the spin they sleep.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace llm {

class WorkerPool {
public:
    using RangeFn = std::function<void(size_t begin, size_t end)>;

    // n_threads counts the calling thread. 0 picks the hardware concurrency.
    explicit WorkerPool(size_t n_threads = 0);
    ~WorkerPool();

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    // Run fn over [0, n) in chunks of `grain`. Blocks until every chunk is done.
    // Not reentrant: call from one thread at a time.
    void parallel_for(size_t n, size_t grain, const RangeFn& fn);

    size_t size() const { return workers_.size() + 1; }

private:
    void worker_loop();
    void run_chunks();

    std::vector<std::thread> workers_;

    std::mutex mtx_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    std::atomic<uint64_t> generation_{0};
    std::atomic<size_t> pending_{0};
    std::atomic<bool> stop_{false};

    // Current job (written before generation_ is bumped).
    const RangeFn* fn_ = nullptr;
    size_t n_ = 0;
    size_t grain_ = 1;
    std::atomic<size_t> next_{0};
};

}  // namespace llm
