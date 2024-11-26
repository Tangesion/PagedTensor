#pragma once
#include "runtime/Tensor.h"
#include "runtime/Buffer.h"
#include "runtime/BufferManager.h"
#include "common/assert.h"
#include <cstdlib>

namespace inference_frame::func
{

    runtime::Tensor::SharedPtr createTensor(std::initializer_list<runtime::Tensor::DimType64> const &dims_list, runtime::Tensor::DataType const &type, runtime::MemoryType device)
    {

        try
        {
            auto dims = runtime::Tensor::makeShape(dims_list);
            runtime::Tensor::SharedPtr tensor = nullptr;
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

    runtime::Tensor::SharedPtr randTensor(std::initializer_list<runtime::Tensor::DimType64> const &dims_list, runtime::Tensor::DataType const &type, runtime::MemoryType device)
    {
        runtime::Tensor::SharedPtr tensor = createTensor(dims_list, type, device);
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

}