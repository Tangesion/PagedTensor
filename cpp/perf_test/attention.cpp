#include <chrono>
#include "func/func.h"
#include "kernel/launch/attention.h"

using MemoryType = paged_tensor::runtime::MemoryType;
using AttentionType = paged_tensor::kernel::cpu::AttentionType;
using DataType = paged_tensor::runtime::Tensor::DataType;
using namespace paged_tensor::kernel::launch;
using namespace paged_tensor::func;

int main()
{
    UniquePtr out = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU, true);
    UniquePtr query = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU, true);
    UniquePtr key = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU, true);
    UniquePtr value = randTensor({1, 32, 1024, 128}, DataType::kFLOAT, MemoryType::kCPU, true);
    UniquePtr interAttn = createTensor({1, 32, 1024, 1024}, DataType::kFLOAT, MemoryType::kCPU, true);
    auto start = std::chrono::high_resolution_clock::now();
    attentionForward(out, query, key, value, interAttn, true, AttentionType::kAttentionOneThread);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "paged prefill one thread : " << duration.count() << " seconds" << std::endl;
}