#include "runtime/bufferManager.h"
#include "runtime/tensor.h"
#include "func/func.h"
#include "func/threadPool.h"
#include "kernel/cpu/matmul.h"
#include <fstream>
// #include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include "kernel/launch/matmul.h"

using namespace toy::runtime;

// int main()
//{
//     // Tensor::randTensor({16, 1, 4}, dataType);
//
//     Tensor::UniquePtr tensor = toy::func::randTensor({2, 2, 3, 4}, DataType::kFLOAT, MemoryType::kCPU);
//     std::cout << *tensor << std::endl;
//     // tensor->reshape({4, 3, 4});
//     toy::func::reShape(tensor, {5, 3, 4});
//     std::cout << *tensor << std::endl;
//
//     // std::ifstream f("../config/support.json");
//     // nlohmann::json data = nlohmann::json::parse(f);
//     // std::cout << data.dump(4) << std::endl;
//     // std::cout << data["dataType"][0] << std::endl;
//     //  auto dims = Tensor::makeShape({16, 1, 4});
//     //  auto constexpr dataType = DataType::kFLOAT;
//     //  Tensor::UniquePtr tensor{BufferManager::cpu(dims, dataType)};
//     //  tensor->printShape();
// }

TEST(MatmulTest, oneThreadTestTime)
{
    toy::kernel::cpu::MatmulType matmulType = toy::kernel::cpu::MatmulType::kMatmulOneThread;
    Tensor::UniquePtr inp = toy::func::randTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    Tensor::UniquePtr weight = toy::func::randTensor({4096, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    // Tensor::UniquePtr bias = nullptr;
    Tensor::UniquePtr out = toy::func::createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    // toy::kernel::cpu::matmulWeightLaunch(out, inp, weight, nullptr, matmulType);
    toy::kernel::launch::matmulWeight(out, inp, weight, nullptr, matmulType);
}

TEST(MatmulTest, threadPoolTestTime)
{
    Tensor::UniquePtr inp = toy::func::randTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    Tensor::UniquePtr weight = toy::func::randTensor({4096, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    // Tensor::UniquePtr bias = nullptr;
    Tensor::UniquePtr out = toy::func::createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    // toy::kernel::cpu::matmulWeightLaunch(out, inp, weight, nullptr, toy::kernel::cpu::MatmulType::kMatmulThreadPool);
    toy::kernel::launch::matmulWeight(out, inp, weight, nullptr, toy::kernel::cpu::MatmulType::kMatmulThreadPool);
}

TEST(MatmulTest, multiThreadTestTime)
{
    Tensor::UniquePtr inp = toy::func::randTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    Tensor::UniquePtr weight = toy::func::randTensor({4096, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    // Tensor::UniquePtr bias = nullptr;
    Tensor::UniquePtr out = toy::func::createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    // toy::kernel::cpu::matmulWeightLaunch(out, inp, weight, nullptr, toy::kernel::cpu::MatmulType::KMatmulMultiThread);
    toy::kernel::launch::matmulWeight(out, inp, weight, nullptr, toy::kernel::cpu::MatmulType::KMatmulMultiThread);
}

// TEST(MatmulTest, equal)
//{
//     Tensor::UniquePtr inp = toy::func::randTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
//     Tensor::UniquePtr weight = toy::func::randTensor({4096, 4096}, DataType::kFLOAT, MemoryType::kCPU);
//     // Tensor::UniquePtr bias = nullptr;
//     Tensor::UniquePtr out1 = toy::func::createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
//     Tensor::UniquePtr out2 = toy::func::createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
//     toy::kernel::cpu::matmulWeightLaunch(out1, inp, weight, nullptr, toy::kernel::cpu::MatmulType::kMatmulOneThread);
//     toy::kernel::cpu::matmulWeightLaunch(out2, inp, weight, nullptr, toy::kernel::cpu::MatmulType::KMatmulMultiThread);
//     auto *data1 = toy::func::getData<Tensor::DataType::kFLOAT>(out1);
//     auto *data2 = toy::func::getData<Tensor::DataType::kFLOAT>(out2);
//     for (int i = 0; i < out1->getSize(); i++)
//     {
//         ASSERT_EQ(data1[i], data2[i]);
//     }
// }