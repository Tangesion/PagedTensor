#include <iostream>
#include <vector>
#include <chrono>
#include <stdlib.h>

int main()
{
    constexpr size_t N = 32 * 4096;
    constexpr size_t poolSize = 4 * 32 * 4096;
    constexpr size_t blockSize = 1024;
    constexpr size_t M = N / blockSize;

    // 测试普通张量
    auto start = std::chrono::high_resolution_clock::now();
    float *base = (float *)malloc(sizeof(float) * N);
    free(base);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "norm tensor malloc and delete time: " << duration.count() << " seconds" << std::endl;

    // 测试分页张量 (优化版本)
    float *memoryPool = (float *)malloc(sizeof(float) * poolSize);

    start = std::chrono::high_resolution_clock::now();
    std::vector<float *> buffer;
    buffer.reserve(M); // 预分配容量
    for (int i = 0; i < M; i++)
    {
        buffer.emplace_back(memoryPool + i * blockSize);
    }
    buffer.clear();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "paged tensor malloc and delete time (optimized): " << duration.count() << " seconds" << std::endl;

    free(memoryPool);
    return 0;
}