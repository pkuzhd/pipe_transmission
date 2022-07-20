//
// Created by zhanghaodan on 2022/7/12.
//

#ifndef RENDERING_THREADPOOL_H
#define RENDERING_THREADPOOL_H

#include <functional>
#include <thread>
#include <vector>

inline int getThreadCount() {
    return std::max<int>(1, std::thread::hardware_concurrency());
}

struct ThreadPool {
    ThreadPool(const int maxThreadsFlag) {
        maxThreads = ThreadPool::getThreadCountFromFlag(maxThreadsFlag);
    }

    ThreadPool() {
        maxThreads = ThreadPool::getThreadCountFromFlag(-1);
    }

    static int getThreadCountFromFlag(const int maxThreadsFlag) {
        return (maxThreadsFlag < 0) ? getThreadCount() : maxThreadsFlag;
    }

    int getMaxThreads() {
        return maxThreads;
    }

    template<class Fn, class... Args>
    void spawn(Fn &&fn, Args &&... args) {
        if (maxThreads == 0) {
            fn(std::forward<Args>(args)...);
        } else {
            if (int(threads.size()) == maxThreads) {
                join();
            }
            threads.emplace_back(std::forward<Fn>(fn), std::forward<Args>(args)...);
        }
    }

    void join() {
        for (std::thread &thread: threads) {
            thread.join();
        }
        threads.clear();
    }

private:
    int maxThreads;
    std::vector<std::thread> threads;
};

#endif //RENDERING_THREADPOOL_H
