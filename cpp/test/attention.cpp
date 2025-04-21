#include <gtest/gtest.h>
#include <chrono>
#include <func/func.h>
#include "kernel/launch/attention.h"

using MemoryType = paged_tensor::runtime::MemoryType;
using AttentionType = paged_tensor::kernel::cpu::AttentionType;
using DataType = paged_tensor::runtime::Tensor::DataType;
using namespace paged_tensor::kernel::launch;
using namespace paged_tensor::func;

TEST(AttentionTest, DISABLED_multiThreadTestTime)
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

TEST(AttentionTest, DISABLED_oneThreadTestTime)
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

TEST(AttentionTest, DISABLED_decodeMultiThreadTestTime)
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

TEST(AttentionTest, DISABLED_decodeOneThreadTestTime)
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

TEST(AttentionTest, DISABLED_multiThreadTestPrint)
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

TEST(AttentionTest, pagedAttentionTestTime)
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
    std::cout << "no paged prefill one thread : " << duration.count() << " seconds" << std::endl;

    out = randTensor({1, 2, 32, 128}, DataType::kFLOAT, MemoryType::kCPU, false);
    query = randTensor({1, 2, 32, 128}, DataType::kFLOAT, MemoryType::kCPU, false);
    key = randTensor({1, 2, 32, 128}, DataType::kFLOAT, MemoryType::kCPU, true);
    value = randTensor({1, 2, 32, 128}, DataType::kFLOAT, MemoryType::kCPU, true);
    interAttn = createTensor({1, 32, 2, 2}, DataType::kFLOAT, MemoryType::kCPU, false);
    start = std::chrono::high_resolution_clock::now();
    attentionForward(out, query, key, value, interAttn, true, AttentionType::kAttentionOneThread);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "paged prefill one thread : " << duration.count() << " seconds" << std::endl;
    std::cout << *out << std::endl;

    /* out = randTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    query = randTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    key = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    value = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    interAttn = createTensor({1, 32, 1, 1024}, DataType::kFLOAT, MemoryType::kCPU);
    start = std::chrono::high_resolution_clock::now();
    attentionForward(out, query, key, value, interAttn, false, AttentionType::kAttentionOneThread);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "np paged decode one thread : " << duration.count() << " seconds" << std::endl;

    out = randTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU, true);
    query = randTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU, true);
    key = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU, true);
    value = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU, true);
    interAttn = createTensor({1, 32, 1, 1024}, DataType::kFLOAT, MemoryType::kCPU, true);
    start = std::chrono::high_resolution_clock::now();
    attentionForward(out, query, key, value, interAttn, false, AttentionType::kAttentionOneThread);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "paged decode one thread : " << duration.count() << " seconds" << std::endl; */
}