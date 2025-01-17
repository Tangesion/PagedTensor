#include <gtest/gtest.h>
#include <func/func.h>

using namespace inference_frame::runtime;
using namespace inference_frame::func;

TEST(TensorTest, wrapTest)
{
    Tensor::UniquePtr tensor = inference_frame::func::randTensor({1, 3, 4}, DataType::kFLOAT, MemoryType::kCPU);
    std::cout << *tensor << std::endl;
    Tensor::UniquePtr tensorWrap = Tensor::wrap(tensor->data(), tensor->getDataType(), tensor->getShape(), tensor->getSize());
    std::cout << *tensorWrap << std::endl;
}