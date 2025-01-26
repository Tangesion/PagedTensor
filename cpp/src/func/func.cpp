#include "func/func.h"

namespace toy::func
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

    // runtime::Tensor::UniquePtr torchToToy(torch::Tensor &tensor)
    //{
    //     void *data = tensor.data_ptr();
    //     auto shape = tensor.sizes();
    //     std::vector<int64_t> shapeVec(shape.begin(), shape.end());
    //     runtime::Tensor::Shape shapeToy = runtime::Tensor::makeShape(shapeVec);
    //     runtime::Tensor::UniquePtr toyTensor = runtime::Tensor::wrap(data, runtime::DataType::kFLOAT, shapeToy);
    //     return toyTensor;
    // }
    //
    // torch::Tensor toyToTorch(runtime::Tensor::UniquePtr &tensor)
    //{
    //    auto shape = tensor->getShape();
    //    std::vector<int64_t> shapeVec(shape.d, shape.d + shape.nbDims);
    //    torch::Tensor torchTensor = torch::from_blob(tensor->data(), shapeVec, torch::kFloat);
    //    return torchTensor;
    //}
}
