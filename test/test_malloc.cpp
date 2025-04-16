#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <queue>
#include <chrono>
#include <vector>
int main()
{
    constexpr size_t N = 32 * 4096;
    constexpr size_t poolSize = 4 * 32 * 4096;
    constexpr size_t blockSize = 1024;
    constexpr size_t M = N / blockSize;
    auto start = std::chrono::high_resolution_clock::now();
    float *base = (float *)malloc(sizeof(float) * N);
    free(base);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "norm tensor malloc and delete time: " << duration.count() << " seconds" << std::endl;

    float *memoryPool = (float *)malloc(sizeof(float) * poolSize);

    start = std::chrono::high_resolution_clock::now();
    std::vector<float *> buffer;
    for (int i = 0; i < M; i++)
    {
        float *curPtr = memoryPool + i * blockSize;
        buffer.emplace_back(curPtr);
    }
    buffer.clear();
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "paged tensor mallo and delete time: " << duration.count() << " seconds" << std::endl;
    return 0;
}