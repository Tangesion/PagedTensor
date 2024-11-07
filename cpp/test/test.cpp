#include "runtime/BufferManager.h"
#include "runtime/Tensor.h"

using namespace inference_frame::runtime;

int main()
{
    auto dims = Tensor::makeShape({16, 1, 4});
    auto constexpr dataType = DataType::kFLOAT;
    Tensor::SharedPtr tensor{BufferManager::cpu(dims, dataType)};
    tensor->printShape();
}