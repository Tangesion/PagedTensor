#include <gtest/gtest.h>
#include <func/func.h>
#include <chrono>
#include "kernel/launch/ffn.h"

using MemoryType = toy::runtime::MemoryType;
using DataType = toy::runtime::Tensor::DataType;
using namespace toy::kernel::launch;
using namespace toy::func;

// TEST(ffnTest, ffnOneThreadPrefillTime)
//{
//     UniquePtr inp = randTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
//     UniquePtr gateProj = randTensor({11008, 4096}, DataType::kFLOAT, MemoryType::kCPU);
//     UniquePtr upProj = randTensor({11008, 4096}, DataType::kFLOAT, MemoryType::kCPU);
//     UniquePtr downProj = randTensor({4096, 11008}, DataType::kFLOAT, MemoryType::kCPU);
//     UniquePtr out = createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
//     auto start = std::chrono::high_resolution_clock::now();
//     ffnForward(out, inp, gateProj, upProj, downProj, false);
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> duration = end - start;
//     std::cout << "Single thread execution time: " << duration.count() << " seconds" << std::endl;
// }

TEST(ffnTest, ffnMultiThreadPrefillTime)
{
    UniquePtr inp = randTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr gateProj = randTensor({11008, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr upProj = randTensor({11008, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr downProj = randTensor({4096, 11008}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr out = createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    ffnForward(out, inp, gateProj, upProj, downProj, true);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Multithread execution time: " << duration.count() << " seconds" << std::endl;
}

TEST(ffnTest, ffnOneThreadDecodeTime)
{
    UniquePtr inp = randTensor({1, 1, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr gateProj = randTensor({11008, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr upProj = randTensor({11008, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr downProj = randTensor({4096, 11008}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr out = createTensor({1, 1, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    ffnForward(out, inp, gateProj, upProj, downProj, false);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Single thread execution time: " << duration.count() << " seconds" << std::endl;
}

TEST(ffnTest, ffnMultiThreadDecodeTime)
{
    UniquePtr inp = randTensor({1, 1, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr gateProj = randTensor({11008, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr upProj = randTensor({11008, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr downProj = randTensor({4096, 11008}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr out = createTensor({1, 1, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    ffnForward(out, inp, gateProj, upProj, downProj, true);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Multithread execution time: " << duration.count() << " seconds" << std::endl;
}
