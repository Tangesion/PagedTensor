#include "func/func.h"

namespace paged_tensor::func
{
    runtime::Tensor::UniquePtr createTensor(std::initializer_list<runtime::Tensor::DimType64> const &dims_list, runtime::Tensor::DataType const &type, runtime::MemoryType device, bool paged)
    {

        try
        {
            auto dims = runtime::Tensor::makeShape(dims_list);
            runtime::Tensor::UniquePtr tensor = nullptr;
            switch (device)
            {
            case runtime::MemoryType::kCPU:
            {
                if (paged)
                {
                    tensor = runtime::BufferManager::cpuPaged(dims, type);
                    break;
                }
                else
                {
                    tensor = runtime::BufferManager::cpu(dims, type);
                    break;
                }
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

    runtime::Tensor::UniquePtr createTensor(runtime::Tensor::Shape const &dims, runtime::Tensor::DataType const &type, runtime::MemoryType device, bool paged)
    {

        try
        {
            runtime::Tensor::UniquePtr tensor = nullptr;
            switch (device)
            {
            case runtime::MemoryType::kCPU:
            {
                if (paged)
                {
                    tensor = runtime::BufferManager::cpuPaged(dims, type);
                    break;
                }
                else
                {
                    tensor = runtime::BufferManager::cpu(dims, type);
                    break;
                }
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

    runtime::Tensor::UniquePtr randTensor(std::initializer_list<runtime::Tensor::DimType64> const &dims_list, runtime::Tensor::DataType const &type, runtime::MemoryType device, bool paged)
    {
        runtime::Tensor::UniquePtr tensor = createTensor(dims_list, type, device, paged);
        // std::cout << "Tensor created" << std::endl;
        try
        {
            switch (type)
            {
            case runtime::Tensor::DataType::kFLOAT:
            {
                if (paged)
                {
                    auto dataPtr = tensor->dataPaged();
                    for (int i = 0; i < tensor->getSize(); i++)
                    {
                        auto tempPtr = dataPtr + i;
                        auto *data = tempPtr.data<float>();
                        *data = static_cast<float>(rand() % 100);
                    }
                    break;
                }
                else
                {
                    auto *data = static_cast<float *>(tensor->data());
                    for (int i = 0; i < tensor->getSize(); i++)
                    {
                        data[i] = static_cast<float>(rand() % 100);
                    }
                    break;
                }
            }
            case runtime::Tensor::DataType::kINT64:
            {
                if (paged)
                {
                    auto dataPtr = tensor->dataPaged();
                    for (int i = 0; i < tensor->getSize(); i++)
                    {
                        auto tempPtr = dataPtr + i;
                        auto *data = tempPtr.data<int64_t>();
                        *data = static_cast<int64_t>(rand() % 100);
                    }
                    break;
                }
                else
                {
                    auto *data = static_cast<int64_t *>(tensor->data());
                    for (int i = 0; i < tensor->getSize(); i++)
                    {
                        data[i] = static_cast<int64_t>(rand() % 100);
                    }
                    break;
                }
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

    runtime::Tensor::UniquePtr makeRange(const int64_t start, const int64_t end, const int64_t span, runtime::MemoryType device)
    {

        if (end <= start)
        {
            throw std::invalid_argument("End must be greater than start.");
        }
        if (span <= 0)
        {
            throw std::invalid_argument("Span must be a positive integer.");
        }

        std::initializer_list<runtime::Tensor::DimType64> dims_list = {static_cast<runtime::Tensor::DimType64>((end - start + span - 1) / span)};
        runtime::Tensor::UniquePtr tensor = createTensor(dims_list, runtime::Tensor::DataType::kINT64, device);

        auto *data = static_cast<int64_t *>(tensor->data());
        for (size_t i = 0; i < tensor->getSize(); i++)
        {
            data[i] = start + i * span;
        }

        return tensor;
    }

    runtime::Tensor::UniquePtr pagedToContinuous(runtime::Tensor::UniquePtr &pagedTensor)
    {
        size_t size = pagedTensor->getSize();
        runtime::Tensor::Shape dims = pagedTensor->getShape();
        runtime::MemoryType device = pagedTensor->getMemoryType();
        runtime::Tensor::DataType type = pagedTensor->getDataType();

        runtime::Tensor::UniquePtr continuousTensor = createTensor(dims, type, device, false);
        DataPtr dataPaged = pagedTensor->dataPaged();

        // now only consider float
        float *data = static_cast<float *>(continuousTensor->data());

        // copy to continuous tensor
        for (size_t i = 0; i < continuousTensor->getSize(); i++)
        {
            data[i] = *(dataPaged[i].data<float>());
        }
        return continuousTensor;
    }
    runtime::Tensor::UniquePtr continuousToPaged(runtime::Tensor::UniquePtr &continuousTensor)
    {
        size_t size = continuousTensor->getSize();
        runtime::Tensor::Shape dims = continuousTensor->getShape();
        runtime::MemoryType device = continuousTensor->getMemoryType();
        runtime::Tensor::DataType type = continuousTensor->getDataType();

        runtime::Tensor::UniquePtr pagedTensor = createTensor(dims, type, device, true);
        DataPtr dataPaged = pagedTensor->dataPaged();
        float *data = static_cast<float *>(continuousTensor->data());

        for (size_t i = 0; i < pagedTensor->getSize(); i++)
        {
            *(dataPaged[i].data<float>()) = data[i];
        }
        return pagedTensor;
    }

    // runtime::Tensor::UniquePtr torchTopaged_tensor(torch::Tensor &tensor)
    //{
    //     void *data = tensor.data_ptr();
    //     auto shape = tensor.sizes();
    //     std::vector<int64_t> shapeVec(shape.begin(), shape.end());
    //     runtime::Tensor::Shape shapepaged_tensor = runtime::Tensor::makeShape(shapeVec);
    //     runtime::Tensor::UniquePtr paged_tensorTensor = runtime::Tensor::wrap(data, runtime::DataType::kFLOAT, shapepaged_tensor);
    //     return paged_tensorTensor;
    // }
    //
    // torch::Tensor paged_tensorToTorch(runtime::Tensor::UniquePtr &tensor)
    //{
    //    auto shape = tensor->getShape();
    //    std::vector<int64_t> shapeVec(shape.d, shape.d + shape.nbDims);
    //    torch::Tensor torchTensor = torch::from_blob(tensor->data(), shapeVec, torch::kFloat);
    //    return torchTensor;
    //}
}
