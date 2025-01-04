#include <gtest/gtest.h>
#include <chrono>
#include "kernel/launch/attention.h"

using MemoryType = inference_frame::runtime::MemoryType;
using AttentionType = inference_frame::kernel::cpu::AttentionType;
using namespace inference_frame::kernel::launch;
using namespace inference_frame::func;

TEST(AttentionTest, multiThreadTestTime)
{
    SharedPtr out = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr query = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr key = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr value = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr interAttn = createTensor({1, 32, 1024, 1024}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    attentionForward(out, query, key, value, interAttn, true, AttentionType::kAttentionMultiThread);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}

TEST(AttentionTest, oneThreadTestTime)
{
    SharedPtr out = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr query = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr key = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr value = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr interAttn = createTensor({1, 32, 1024, 1024}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    attentionForward(out, query, key, value, interAttn, true, AttentionType::kAttentionOneThread);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}

TEST(AttentionTest, decodeMultiThreadTestTime)
{
    SharedPtr out = randTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr query = randTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr key = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr value = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr interAttn = createTensor({1, 32, 1, 1024}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    attentionForward(out, query, key, value, interAttn, false, AttentionType::kAttentionMultiThread);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}

TEST(AttentionTest, decodeOneThreadTestTime)
{
    SharedPtr out = randTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr query = randTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr key = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr value = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr interAttn = createTensor({1, 32, 1, 1024}, DataType::kFLOAT, MemoryType::kCPU);
    auto start = std::chrono::high_resolution_clock::now();
    attentionForward(out, query, key, value, interAttn, false, AttentionType::kAttentionOneThread);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}

TEST(AttentionTest, equal)
{
    SharedPtr out = randTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr query = randTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr key = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr value = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    SharedPtr interAttn = createTensor({1, 32, 1024, 1024}, DataType::kFLOAT, MemoryType::kCPU);
    attentionForward(out, query, key, value, interAttn, false, AttentionType::kAttentionMultiThread);
    SharedPtr outOneThread = createTensor({1, 32, 1, 128}, DataType::kFLOAT, MemoryType::kCPU);
    attentionForward(outOneThread, query, key, value, interAttn, false, AttentionType::kAttentionOneThread);
    auto *outData = getData<DataType::kFLOAT>(out);
    auto *outOneThreadData = getData<DataType::kFLOAT>(outOneThread);
    for (size_t i = 0; i < out->getShape().d[0] * out->getShape().d[1] * out->getShape().d[2] * out->getShape().d[3]; i++)
    {
        EXPECT_EQ(outData[i], outOneThreadData[i]);
    }
}
