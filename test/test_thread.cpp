#include <iostream>
#include <thread>
#include <condition_variable>
#include <mutex>

std::condition_variable cv;
std::mutex mtx;

void test()
{
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock); // 等待条件变量的通知
    std::cout << "Thread finished execution" << std::endl;
}

int main()
{
    std::thread t(test);
    t.detach(); // 分离线程，后台任务独立运行

    // std::this_thread::sleep_for(std::chrono::seconds(3)); // 确保主线程不会过早退出

    return 0;
}