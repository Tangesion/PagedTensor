#pragma once
#include "runtime/tensor.h"
#include "runtime/buffer.h"
#include "runtime/bufferManager.h"
#include "common/assert.h"
#include <cstdlib>
#include <variant>

namespace inference_frame::func
{

    runtime::Tensor::UniquePtr createTensor(std::initializer_list<runtime::Tensor::DimType64> const &dims_list, runtime::Tensor::DataType const &type, runtime::MemoryType device)
    {

        try
        {
            auto dims = runtime::Tensor::makeShape(dims_list);
            runtime::Tensor::UniquePtr tensor = nullptr;
            switch (device)
            {
            case runtime::MemoryType::kCPU:
            {
                tensor = runtime::BufferManager::cpu(dims, type);
                break;
            }
            default:
            {
                JUST_THROW("Unsupported memory type");
                break;
            }
            }
            return tensor;
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            std::exit(EXIT_FAILURE);
        }
    }

    runtime::Tensor::UniquePtr randTensor(std::initializer_list<runtime::Tensor::DimType64> const &dims_list, runtime::Tensor::DataType const &type, runtime::MemoryType device)
    {
        runtime::Tensor::UniquePtr tensor = createTensor(dims_list, type, device);
        // std::cout << "Tensor created" << std::endl;
        try
        {
            switch (type)
            {
            case runtime::Tensor::DataType::kFLOAT:
            {
                auto *data = static_cast<float *>(tensor->data());
                for (int i = 0; i < tensor->getSize(); i++)
                {
                    data[i] = static_cast<float>(rand()) / RAND_MAX;
                }
                break;
            }
            case runtime::Tensor::DataType::kINT64:
            {
                auto *data = static_cast<int64_t *>(tensor->data());
                for (int i = 0; i < tensor->getSize(); i++)
                {
                    data[i] = rand() % 100;
                }
                break;
            }
            default:
            {
                JUST_THROW("Unsupported data type");
                break;
            }
            }
            return tensor;
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            std::exit(EXIT_FAILURE);
        }
    }
    void reShape(runtime::Tensor::UniquePtr &tensor, std::initializer_list<runtime::Tensor::DimType64> const &dims_list)
    {
        try
        {
            std::size_t size = tensor->getSize();
            runtime::Tensor::Shape dims = runtime::Tensor::makeShape(dims_list);
            CHECK_WITH_INFO(size == runtime::Tensor::volume(dims), "New shape size must be equal to the original size");

            tensor->reshape(dims);
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            std::exit(EXIT_FAILURE);
        }
    }

    // template <runtime::Tensor::DataType T>
    // struct DataTypeInfo;

    // template <>
    // struct DataTypeInfo<runtime::Tensor::DataType::kFLOAT>
    // {
    //     using Type = float;
    // };

    // template <>
    // struct DataTypeInfo<runtime::Tensor::DataType::kINT32>
    // {
    //     using Type = int32_t;
    // };

    // using TensorDataVariant = std::variant<float *, int32_t *>;

    // template <runtime::Tensor::DataType T>
    // typename DataTypeInfo<T>::Type *getData(runtime::Tensor::UniquePtr tensor)
    // {
    //     return static_cast<typename DataTypeInfo<T>::Type *>(tensor->data());
    // }

    // TensorDataVariant getData(runtime::Tensor::UniquePtr tensor)
    // {
    //     switch (tensor->getDataType())
    //     {
    //     case runtime::Tensor::DataType::kFLOAT:
    //         return getData<runtime::Tensor::DataType::kFLOAT>(tensor);
    //     case runtime::Tensor::DataType::kINT32:
    //         return getData<runtime::Tensor::DataType::kINT32>(tensor);
    //     default:
    //         JUST_THROW("Unsupported data type");
    //         break;
    //     }
    // }

}