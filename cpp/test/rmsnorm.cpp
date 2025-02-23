#include <gtest/gtest.h>
#include <func/func.h>
#include <chrono>
#include "kernel/launch/rmsnorm.h"

using MemoryType = paged_tensor::runtime::MemoryType;
using DataType = paged_tensor::runtime::Tensor::DataType;
using namespace paged_tensor::kernel::launch;
using namespace paged_tensor::func;

TEST(rmsNormTest, muliThreadPrefillTestTime)
{
    UniquePtr out = createTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr inp = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr weight = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    rmsNorm(out, inp, weight, true);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}
TEST(rmsNormTest, oneThreadPrefillTestTime)
{
    UniquePtr out = createTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr inp = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr weight = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    rmsNorm(out, inp, weight, false);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}
TEST(rmsNormTest, muliThreadDecodeTestTime)
{
    UniquePtr out = createTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr inp = randTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr weight = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    rmsNorm(out, inp, weight, true);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}
TEST(rmsNormTest, oneThreadDecodeTestTime)
{
    UniquePtr out = createTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr inp = randTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr weight = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    rmsNorm(out, inp, weight, false);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}
