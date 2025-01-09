#include <gtest/gtest.h>
#include <chrono>
#include <func/func.h>
#include "kernel/launch/attention.h"

using MemoryType = inference_frame::runtime::MemoryType;
using AttentionType = inference_frame::kernel::cpu::AttentionType;
using namespace inference_frame::kernel::launch;
using namespace inference_frame::func;

TEST(AttentionTest, multiThreadTestTime)
{
    UniquePtr out = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr query = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr key = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr value = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr interAttn = createTensor({1, 32, 1024, 1024}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    attentionForward(out, query, key, value, interAttn, true, AttentionType::kAttentionMultiThread);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}

TEST(AttentionTest, oneThreadTestTime)
{
    UniquePtr out = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr query = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr key = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr value = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr interAttn = createTensor({1, 32, 1024, 1024}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    attentionForward(out, query, key, value, interAttn, true, AttentionType::kAttentionOneThread);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}

TEST(AttentionTest, decodeMultiThreadTestTime)
{
    UniquePtr out = randTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr query = randTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr key = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr value = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr interAttn = createTensor({1, 32, 1, 1024}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    attentionForward(out, query, key, value, interAttn, false, AttentionType::kAttentionMultiThread);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}

// TEST(AttentionTest, FP16)
//{
//     UniquePtr out = randTensor({1, 32, 1, 128}, DataType::kHALF, MemoryType::kCPU);
//     UniquePtr query = randTensor({1, 32, 1, 128}, DataType::kHALF, MemoryType::kCPU);
//     UniquePtr key = randTensor({1, 32, 1024, 128}, DataType::kHALF, MemoryType::kCPU);
//     UniquePtr value = randTensor({1, 32, 1024, 128}, DataType::kHALF, MemoryType::kCPU);
//     UniquePtr interAttn = createTensor({1, 32, 1024, 1024}, DataType::kHALF, MemoryType::kCPU);
//     auto start = std::chrono::high_resolution_clock::now();
//     attentionForward(out, query, key, value, interAttn, false, AttentionType::kAttentionMultiThread);
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> duration = end - start;
//     std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
// }

TEST(AttentionTest, decodeOneThreadTestTime)
{
    UniquePtr out = randTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr query = randTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr key = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr value = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr interAttn = createTensor({1, 32, 1, 1024}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    attentionForward(out, query, key, value, interAttn, false, AttentionType::kAttentionOneThread);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}
