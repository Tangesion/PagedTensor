#include <gtest/gtest.h>
#include "func/func.h"
#include <chrono>

using namespace paged_tensor::runtime;
using namespace paged_tensor::func;

TEST(TensorTest, wrapTest)
{
    Tensor::UniquePtr tensor = paged_tensor::func::randTensor({1, 3, 4}, DataType::kFLOAT, MemoryType::kCPU);
    std::cout << *tensor << std::endl;
    Tensor::UniquePtr tensorWrap = Tensor::wrap(tensor->data(), tensor->getDataType(), tensor->getShape(), tensor->getSize());
    std::cout << *tensorWrap << std::endl;
}

TEST(TensorTest, pagedTensorTest)
{
    // block size 16  block num 3
    Tensor::UniquePtr tensor1 = paged_tensor::func::randTensor({1, 3, 4}, DataType::kFLOAT, MemoryType::kCPU, true);
    std::cout << *tensor1 << std::endl;
    Tensor::UniquePtr tensor2 = paged_tensor::func::randTensor({1, 3, 4}, DataType::kFLOAT, MemoryType::kCPU, true);
    std::cout << *tensor2 << std::endl;
    Tensor::UniquePtr tensor3 = paged_tensor::func::randTensor({1, 3, 4}, DataType::kFLOAT, MemoryType::kCPU, true);
    std::cout << *tensor3 << std::endl;

    tensor1.reset();
    tensor3.reset();
    Tensor::UniquePtr tensor4 = paged_tensor::func::randTensor({1, 3, 8}, DataType::kFLOAT, MemoryType::kCPU, true);
    std::cout << *tensor4 << std::endl;
}

TEST(TensorTest, pagedTensorExtendTest)
{
    // block size 16  block num 3
    Tensor::UniquePtr tensor1 = paged_tensor::func::randTensor({1, 3, 4}, DataType::kFLOAT, MemoryType::kCPU, true);
}

TEST(TensorTest, pagedToContinuousTest)
{
    Tensor::UniquePtr tensor1 = paged_tensor::func::randTensor({1, 3, 4}, DataType::kFLOAT, MemoryType::kCPU, true);
    std::cout << *tensor1 << std::endl;
    Tensor::UniquePtr tensor2 = paged_tensor::func::pagedToContinuous(tensor1);
    std::cout << *tensor2 << std::endl;
    Tensor::UniquePtr tensor3 = paged_tensor::func::continuousToPaged(tensor2);
    std::cout << *tensor3 << std::endl;
}

TEST(TensorTest, pagedTensorTimeTest)
{
    try
    {
        std::initializer_list<Tensor::DimType64> const &dims_list = {1, 3, 13};
        auto dims = Tensor::makeShape(dims_list);
        MemoryType device = MemoryType::kCPU;
        Tensor::UniquePtr tensor = nullptr;
        Tensor::DataType type = Tensor::DataType::kFLOAT;
        switch (device)
        {
        case MemoryType::kCPU:
        {
            tensor = BufferManager::cpuPaged(dims, type);
            std::cout << "Paged tensor created!" << std::endl;

            break;
        }
        default:
        {
            JUST_THROW("Unsupported memory type");
            break;
        }
        }
        DataPtr dataPtr = tensor->dataPaged();
        for (int i = 0; i < tensor->getSize(); i++)
        {
            // DataPtr tempPtr = dataPtr + i;
            // auto *data = static_cast<float *>(tempPtr.data());
            //*data = i;
            auto *data = (dataPtr[i].data<float>());
            *data = i;
        }
        dataPtr = tensor->dataPaged();
        for (int i = 0; i < tensor->getSize(); i++)
        {
            // DataPtr tempPtr = dataPtr + i;
            auto *data = static_cast<float *>(dataPtr[i].data<float>());
            std::cout << *data << " ";
        }
        std::cout << std::endl;
        std::cout << *tensor << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        dataPtr = tensor->dataPaged();
        for (int i = 0; i < tensor->getSize(); i++)
        {
            // DataPtr tempPtr = dataPtr + i;
            auto *data = static_cast<float *>(dataPtr[i].data<float>());
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "paged read time: " << duration.count() << " seconds" << std::endl;

        float *data2 = new float[tensor->getSize()];
        for (int i = 0; i < tensor->getSize(); i++)
        {
            data2[i] = i;
        }
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < tensor->getSize(); i++)
        {
            auto *data = static_cast<float *>(dataPtr[i].data<float>());
        }
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        std::cout << "normal read time: " << duration.count() << " seconds" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        std::exit(EXIT_FAILURE);
    }
}