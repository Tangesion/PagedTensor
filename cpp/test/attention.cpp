#include <gtest/gtest.h>
#include <chrono>
#include <func/func.h>
#include "kernel/launch/attention.h"

using MemoryType = paged_tensor::runtime::MemoryType;
using AttentionType = paged_tensor::kernel::cpu::AttentionType;
using DataType = paged_tensor::runtime::Tensor::DataType;
using namespace paged_tensor::kernel::launch;
using namespace paged_tensor::func;

TEST(AttentionTest, multiThreadTestTime)
{
    UniquePtr out = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr query = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr key = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr value = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr interAttn = createTensor({1, 32, 1024, 1024}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    attentionForward(out, query, key, value, interAttn, true, AttentionType::kAttentionMultiThread);
    // std::cout << *out << std::endl;
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

TEST(AttentionTest, multiThreadTestPrint)
{
    UniquePtr out = randTensor({1, 32, 4, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr query = randTensor({1, 32, 4, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr key = randTensor({1, 32, 4, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr value = randTensor({1, 32, 4, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr interAttn = createTensor({1, 32, 4, 4}, DataType::kFLOAT, MemoryType::kCPU);
    attentionForward(out, query, key, value, interAttn, true, AttentionType::kAttentionMultiThread);
    std::cout << *interAttn << std::endl;
    // std::cout << *out << std::endl;
}