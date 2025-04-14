
#include "func/func.h"
#include <fstream>
// #include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include "kernel/launch/matmul.h"

using namespace paged_tensor::runtime;

// int main()
//{
//     // Tensor::randTensor({16, 1, 4}, dataType);
//
//     Tensor::UniquePtr tensor = paged_tensor::func::randTensor({2, 2, 3, 4}, DataType::kFLOAT, MemoryType::kCPU);
//     std::cout << *tensor << std::endl;
//     // tensor->reshape({4, 3, 4});
//     paged_tensor::func::reShape(tensor, {5, 3, 4});
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

TEST(MatmulTest, DISABLED_oneThreadTestTime)
{
    paged_tensor::kernel::cpu::MatmulType matmulType = paged_tensor::kernel::cpu::MatmulType::kMatmulOneThread;
    Tensor::UniquePtr inp = paged_tensor::func::randTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    Tensor::UniquePtr weight = paged_tensor::func::randTensor({4096, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    // Tensor::UniquePtr bias = nullptr;
    Tensor::UniquePtr out = paged_tensor::func::createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    // paged_tensor::kernel::cpu::matmulWeightLaunch(out, inp, weight, nullptr, matmulType);
    paged_tensor::kernel::launch::matmulWeight(out, inp, weight, nullptr, matmulType);
}

TEST(MatmulTest, DISABLED_threadPoolTestTime)
{
    Tensor::UniquePtr inp = paged_tensor::func::randTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    Tensor::UniquePtr weight = paged_tensor::func::randTensor({4096, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    // Tensor::UniquePtr bias = nullptr;
    Tensor::UniquePtr out = paged_tensor::func::createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    // paged_tensor::kernel::cpu::matmulWeightLaunch(out, inp, weight, nullptr, paged_tensor::kernel::cpu::MatmulType::kMatmulThreadPool);
    paged_tensor::kernel::launch::matmulWeight(out, inp, weight, nullptr, paged_tensor::kernel::cpu::MatmulType::kMatmulThreadPool);
}

TEST(MatmulTest, DISABLED_multiThreadTestTime)
{
    Tensor::UniquePtr inp = paged_tensor::func::randTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    Tensor::UniquePtr weight = paged_tensor::func::randTensor({4096, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    // Tensor::UniquePtr bias = nullptr;
    Tensor::UniquePtr out = paged_tensor::func::createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    // paged_tensor::kernel::cpu::matmulWeightLaunch(out, inp, weight, nullptr, paged_tensor::kernel::cpu::MatmulType::kMatmulMultiThread);
    paged_tensor::kernel::launch::matmulWeight(out, inp, weight, nullptr, paged_tensor::kernel::cpu::MatmulType::kMatmulMultiThread);
}

TEST(MatmulTest, pagedTestTime)
{
    paged_tensor::kernel::cpu::MatmulType matmulType = paged_tensor::kernel::cpu::MatmulType::kMatmulOneThread;
    Tensor::UniquePtr inp = paged_tensor::func::randTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    Tensor::UniquePtr weight = paged_tensor::func::randTensor({4096, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    // Tensor::UniquePtr bias = nullptr;
    Tensor::UniquePtr out = paged_tensor::func::createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
    // paged_tensor::kernel::cpu::matmulWeightLaunch(out, inp, weight, nullptr, matmulType);
    auto start = std::chrono::high_resolution_clock::now();
    paged_tensor::kernel::launch::matmulWeight(out, inp, weight, nullptr, matmulType);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "no paged matmul one thread time: " << duration.count() << " seconds" << std::endl;

    // matmulType = paged_tensor::kernel::cpu::MatmulType::kMatmulOneThread;
    // inp = paged_tensor::func::randTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU, true);
    // weight = paged_tensor::func::randTensor({4096, 4096}, DataType::kFLOAT, MemoryType::kCPU, false);
    //// Tensor::UniquePtr bias = nullptr;
    // out = paged_tensor::func::createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU, true);
    //// paged_tensor::kernel::cpu::matmulWeightLaunch(out, inp, weight, nullptr, matmulType);
    // start = std::chrono::high_resolution_clock::now();
    // paged_tensor::kernel::launch::matmulWeight(out, inp, weight, nullptr, matmulType);
    // end = std::chrono::high_resolution_clock::now();
    // duration = end - start;
    // std::cout << "paged matmul one thread time: " << duration.count() << " seconds" << std::endl;

    matmulType = paged_tensor::kernel::cpu::MatmulType::kMatmulBlock;
    inp = paged_tensor::func::randTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU, true);
    weight = paged_tensor::func::randTensor({4096, 4096}, DataType::kFLOAT, MemoryType::kCPU, false);
    // Tensor::UniquePtr bias = nullptr;
    out = paged_tensor::func::createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU, true);
    // paged_tensor::kernel::cpu::matmulWeightLaunch(out, inp, weight, nullptr, matmulType);
    start = std::chrono::high_resolution_clock::now();
    paged_tensor::kernel::launch::matmulWeight(out, inp, weight, nullptr, matmulType);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "paged matmul block one thread time: " << duration.count() << " seconds" << std::endl;

    matmulType = paged_tensor::kernel::cpu::MatmulType::kMatmulInternBlock;
    inp = paged_tensor::func::randTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU, true);
    weight = paged_tensor::func::randTensor({4096, 4096}, DataType::kFLOAT, MemoryType::kCPU, false);
    // Tensor::UniquePtr bias = nullptr;
    out = paged_tensor::func::createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU, true);
    // paged_tensor::kernel::cpu::matmulWeightLaunch(out, inp, weight, nullptr, matmulType);
    start = std::chrono::high_resolution_clock::now();
    paged_tensor::kernel::launch::matmulWeight(out, inp, weight, nullptr, matmulType);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "paged matmul intern block one thread time: " << duration.count() << " seconds" << std::endl;

    matmulType = paged_tensor::kernel::cpu::MatmulType::KMatmulBlockMultiThread;
    inp = paged_tensor::func::randTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU, true);
    weight = paged_tensor::func::randTensor({4096, 4096}, DataType::kFLOAT, MemoryType::kCPU, false);
    // Tensor::UniquePtr bias = nullptr;
    out = paged_tensor::func::createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU, true);
    // paged_tensor::kernel::cpu::matmulWeightLaunch(out, inp, weight, nullptr, matmulType);
    start = std::chrono::high_resolution_clock::now();
    paged_tensor::kernel::launch::matmulWeight(out, inp, weight, nullptr, matmulType);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "paged matmul block multi thread time: " << duration.count() << " seconds" << std::endl;

    matmulType = paged_tensor::kernel::cpu::MatmulType::kMatmulMultiThread;
    inp = paged_tensor::func::randTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU, false);
    weight = paged_tensor::func::randTensor({4096, 4096}, DataType::kFLOAT, MemoryType::kCPU, false);
    // Tensor::UniquePtr bias = nullptr;
    out = paged_tensor::func::createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU, false);
    // paged_tensor::kernel::cpu::matmulWeightLaunch(out, inp, weight, nullptr, matmulType);
    start = std::chrono::high_resolution_clock::now();
    paged_tensor::kernel::launch::matmulWeight(out, inp, weight, nullptr, matmulType);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "matmul multi thread time: " << duration.count() << " seconds" << std::endl;

    matmulType = paged_tensor::kernel::cpu::MatmulType::kMatmulMultiThread;
    inp = paged_tensor::func::randTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU, true);
    weight = paged_tensor::func::randTensor({4096, 4096}, DataType::kFLOAT, MemoryType::kCPU, false);
    // Tensor::UniquePtr bias = nullptr;
    out = paged_tensor::func::createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU, true);
    // paged_tensor::kernel::cpu::matmulWeightLaunch(out, inp, weight, nullptr, matmulType);
    start = std::chrono::high_resolution_clock::now();
    paged_tensor::kernel::launch::matmulWeight(out, inp, weight, nullptr, matmulType);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "paged matmul multi thread time: " << duration.count() << " seconds" << std::endl;
}

// TEST(MatmulTest, equal)
//{
//     Tensor::UniquePtr inp = paged_tensor::func::randTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
//     Tensor::UniquePtr weight = paged_tensor::func::randTensor({4096, 4096}, DataType::kFLOAT, MemoryType::kCPU);
//     // Tensor::UniquePtr bias = nullptr;
//     Tensor::UniquePtr out1 = paged_tensor::func::createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
//     Tensor::UniquePtr out2 = paged_tensor::func::createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU);
//     paged_tensor::kernel::cpu::matmulWeightLaunch(out1, inp, weight, nullptr, paged_tensor::kernel::cpu::MatmulType::kMatmulOneThread);
//     paged_tensor::kernel::cpu::matmulWeightLaunch(out2, inp, weight, nullptr, paged_tensor::kernel::cpu::MatmulType::kMatmulMultiThread);
//     auto *data1 = paged_tensor::func::getData<Tensor::DataType::kFLOAT>(out1);
//     auto *data2 = paged_tensor::func::getData<Tensor::DataType::kFLOAT>(out2);
//     for (int i = 0; i < out1->getSize(); i++)
//     {
//         ASSERT_EQ(data1[i], data2[i]);
//     }
// }