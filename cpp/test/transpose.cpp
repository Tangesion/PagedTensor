#include <gtest/gtest.h>
#include <chrono>
#include "kernel/launch/transpose.h"

using MemoryType = paged_tensor::runtime::MemoryType;
using DataType = paged_tensor::runtime::Tensor::DataType;
using namespace paged_tensor::kernel::launch;
using namespace paged_tensor::func;

TEST(transposeTest, transposeOneThreadPrefillTime)
{
    UniquePtr out = createTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr inp = randTensor({1, 1024, 32, 128}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    transpose(out, inp, false);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Single thread execution time: " << duration.count() << " seconds" << std::endl;
}

TEST(transposeTest, transposeMultiThreadPrefillTime)
{
    UniquePtr out = createTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr inp = randTensor({1, 1024, 32, 128}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    transpose(out, inp, true);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Multithread execution time: " << duration.count() << " seconds" << std::endl;
}

TEST(transposeTest, transposeOneThreadDecodeTime)
{
    UniquePtr out = createTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr inp = randTensor({1, 1, 32, 128}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    transpose(out, inp, false);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Single thread execution time: " << duration.count() << " seconds" << std::endl;
}

TEST(transposeTest, transposeMultiThreadDecodeTime)
{
    UniquePtr out = createTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr inp = randTensor({1, 1, 32, 128}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    transpose(out, inp, true);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Multithread execution time: " << duration.count() << " seconds" << std::endl;
}