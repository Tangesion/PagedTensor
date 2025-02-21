#include <gtest/gtest.h>
#include <func/func.h>

using namespace toy::runtime;
using namespace toy::func;

TEST(TensorTest, wrapTest)
{
    Tensor::UniquePtr tensor = toy::func::randTensor({1, 3, 4}, DataType::kFLOAT, MemoryType::kCPU);
    std::cout << *tensor << std::endl;
    Tensor::UniquePtr tensorWrap = Tensor::wrap(tensor->data(), tensor->getDataType(), tensor->getShape(), tensor->getSize());
    std::cout << *tensorWrap << std::endl;
}

TEST(TensorTest, pagedTensorTest)
{
    try
    {
        std::initializer_list<Tensor::DimType64> const &dims_list = {1, 3, 4};
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
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        std::exit(EXIT_FAILURE);
    }
}