#include "runtime/BufferManager.h"
#include "runtime/Tensor.h"
#include "func/func.h"

using namespace inference_frame::runtime;

int main()
{
    // Tensor::randTensor({16, 1, 4}, dataType);

    Tensor::SharedPtr tensor = inference_frame::func::randTensor({2, 2, 3, 4}, DataType::kFLOAT, MemoryType::kCPU);
    std::cout << *tensor << std::endl;

    // auto dims = Tensor::makeShape({16, 1, 4});
    // auto constexpr dataType = DataType::kFLOAT;
    // Tensor::SharedPtr tensor{BufferManager::cpu(dims, dataType)};
    // tensor->printShape();
}