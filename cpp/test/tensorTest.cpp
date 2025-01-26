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