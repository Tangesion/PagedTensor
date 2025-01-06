#include <gtest/gtest.h>
#include <chrono>
#include "kernel/launch/rmsnorm.h"

using MemoryType = inference_frame::runtime::MemoryType;
using namespace inference_frame::kernel::launch;
using namespace inference_frame::func;

TEST(rmsNormTest, muliThreadPrefillTestTime)
{
    SharedPtr out = createTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr inp = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr weight = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    rmsNorm(out, inp, weight, true);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}
TEST(rmsNormTest, oneThreadPrefillTestTime)
{
    SharedPtr out = createTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr inp = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr weight = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    rmsNorm(out, inp, weight, false);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}
TEST(rmsNormTest, muliThreadDecodeTestTime)
{
    SharedPtr out = createTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr inp = randTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr weight = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    rmsNorm(out, inp, weight, true);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}
TEST(rmsNormTest, oneThreadDecodeTestTime)
{
    SharedPtr out = createTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr inp = randTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr weight = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    rmsNorm(out, inp, weight, false);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}
