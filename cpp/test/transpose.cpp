#include <gtest/gtest.h>
#include <chrono>
#include "kernel/launch/transpose.h"

using MemoryType = inference_frame::runtime::MemoryType;
using namespace inference_frame::kernel::launch;
using namespace inference_frame::func;

TEST(transposeTest, transposeOneThreadPrefillTime)
{
    SharedPtr out = createTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr inp = randTensor({1, 1024, 32, 128}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    transpose(out, inp, false);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Single thread execution time: " << duration.count() << " seconds" << std::endl;
}

TEST(transposeTest, transposeMultiThreadPrefillTime)
{
    SharedPtr out = createTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr inp = randTensor({1, 1024, 32, 128}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    transpose(out, inp, true);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Multithread execution time: " << duration.count() << " seconds" << std::endl;
}

TEST(transposeTest, transposeOneThreadDecodeTime)
{
    SharedPtr out = createTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr inp = randTensor({1, 1, 32, 128}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    transpose(out, inp, false);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Single thread execution time: " << duration.count() << " seconds" << std::endl;
}

TEST(transposeTest, transposeMultiThreadDecodeTime)
{
    SharedPtr out = createTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr inp = randTensor({1, 1, 32, 128}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    transpose(out, inp, true);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Multithread execution time: " << duration.count() << " seconds" << std::endl;
}