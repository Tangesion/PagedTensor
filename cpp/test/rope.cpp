#include <gtest/gtest.h>
#include <func/func.h>
#include <chrono>
#include "kernel/launch/rope.h"

using MemoryType = inference_frame::runtime::MemoryType;
using namespace inference_frame::kernel::launch;
using namespace inference_frame::func;

TEST(ropeTest, precomputeFreqsCosSinTime)
{
    SharedPtr freqsCosSin = createTensor({1024, 2, 128}, DataType::kFLOAT, MemoryType::kCPU);
    auto start2 = std::chrono::high_resolution_clock::now();
    precomputeFreqsCosSin(freqsCosSin, 128, 1024, 10000.0, false);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration2 = end2 - start2;
    std::cout << "Single thread execution time: " << duration2.count() << " seconds" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    precomputeFreqsCosSin(freqsCosSin, 128, 1024, 10000.0, true);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Multithread execution time: " << duration.count() << " seconds" << std::endl;
}

TEST(ropeTest, applyRopeTime)
{
    SharedPtr out = createTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr inp = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr freqsCosSin = randTensor({1024, 2, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr pos = randTensor({1024}, DataType::kINT64, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    applyRope(out, inp, freqsCosSin, pos, true);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Multithread execution time: " << duration.count() << " seconds" << std::endl;
    auto start2 = std::chrono::high_resolution_clock::now();
    applyRope(out, inp, freqsCosSin, pos, false);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration2 = end2 - start2;
    std::cout << "Single thread execution time: " << duration2.count() << " seconds" << std::endl;
}
