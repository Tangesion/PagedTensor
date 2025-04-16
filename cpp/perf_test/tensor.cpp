#include <chrono>
#include "func/func.h"

using MemoryType = paged_tensor::runtime::MemoryType;
using DataType = paged_tensor::runtime::Tensor::DataType;
using UniquePtr = paged_tensor::runtime::Tensor::UniquePtr;
using namespace paged_tensor::func;

int main()
{
    // UniquePtr tensor1 = createTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU);
    UniquePtr tensor2 = createTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU, true);
}