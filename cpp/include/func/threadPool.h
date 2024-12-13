#pragma once
#include <vector>
#include <functional>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace inference_frame::func
{
    class ThreadPool
    {
    public:
        ThreadPool(size_t numThreads) : stop(false)
        {
            for (size_t i = 0; i < numThreads; i++)
            {
                workers.emplace_back([this]
                                     {
                for (;;)
                {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->mtx);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                } });
            }
        }

        ~ThreadPool()
        {
            {
                std::unique_lock<std::mutex> lock(mtx);
                stop = true;
            }
            condition.notify_all();
            for (std::thread &worker : workers)
                worker.join();
        }

        template <class F, class... Args>
        void enqueue(F &&f, Args &&...args)
        {
            auto task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
            {
                std::unique_lock<std::mutex> lock(mtx);
                tasks.emplace(task);
            }
            condition.notify_one();
        }

    private:
        std::vector<std::thread> workers;
        std::queue<std::function<void()>> tasks;

        std::mutex mtx;
        std::condition_variable condition;
        bool stop;
    };
} // namespace inference_frame::func
