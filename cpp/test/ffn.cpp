#include <gtest/gtest.h>
#include <func/func.h>
#include <chrono>
#include "kernel/launch/ffn.h"

using MemoryType = inference_frame::runtime::MemoryType;
using namespace inference_frame::kernel::launch;
using namespace inference_frame::func;

// TEST(ffnTest, ffnOneThreadPrefillTime)
//{
//     SharedPtr inp = randTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
//     SharedPtr gateProj = randTensor({11008, 4096}, DataType::kFLOAT, MemoryType::kCPU);
//     SharedPtr upProj = randTensor({11008, 4096}, DataType::kFLOAT, MemoryType::kCPU);
//     SharedPtr downProj = randTensor({4096, 11008}, DataType::kFLOAT, MemoryType::kCPU);
//     SharedPtr out = createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
//     auto start = std::chrono::high_resolution_clock::now();
//     ffnForward(out, inp, gateProj, upProj, downProj, false);
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> duration = end - start;
//     std::cout << "Single thread execution time: " << duration.count() << " seconds" << std::endl;
// }

TEST(ffnTest, ffnMultiThreadPrefillTime)
{
    SharedPtr inp = randTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr gateProj = randTensor({11008, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr upProj = randTensor({11008, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr downProj = randTensor({4096, 11008}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr out = createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    ffnForward(out, inp, gateProj, upProj, downProj, true);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Multithread execution time: " << duration.count() << " seconds" << std::endl;
}

TEST(ffnTest, ffnOneThreadDecodeTime)
{
    SharedPtr inp = randTensor({1, 1, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr gateProj = randTensor({11008, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr upProj = randTensor({11008, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr downProj = randTensor({4096, 11008}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr out = createTensor({1, 1, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    ffnForward(out, inp, gateProj, upProj, downProj, false);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Single thread execution time: " << duration.count() << " seconds" << std::endl;
}

TEST(ffnTest, ffnMultiThreadDecodeTime)
{
    SharedPtr inp = randTensor({1, 1, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr gateProj = randTensor({11008, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr upProj = randTensor({11008, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr downProj = randTensor({4096, 11008}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr out = createTensor({1, 1, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    ffnForward(out, inp, gateProj, upProj, downProj, true);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Multithread execution time: " << duration.count() << " seconds" << std::endl;
}
