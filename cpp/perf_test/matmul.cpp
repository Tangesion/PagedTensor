#include "func/func.h"
#include "kernel/launch/matmul.h"

using namespace paged_tensor::runtime;

int main()
{
    paged_tensor::kernel::cpu::MatmulType matmulType = paged_tensor::kernel::cpu::MatmulType::kMatmulBlock;
    Tensor::UniquePtr inp = paged_tensor::func::randTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU, true);
    Tensor::UniquePtr weight = paged_tensor::func::randTensor({4096, 4096}, DataType::kFLOAT, MemoryType::kCPU, false);
    // Tensor::UniquePtr bias = nullptr;
    Tensor::UniquePtr out = paged_tensor::func::createTensor({1, 1024, 4096}, DataType::kFLOAT, MemoryType::kCPU, true);
    // paged_tensor::kernel::cpu::matmulWeightLaunch(out, inp, weight, nullptr, matmulType);
    auto start = std::chrono::high_resolution_clock::now();
    paged_tensor::kernel::launch::matmulWeight(out, inp, weight, nullptr, matmulType);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "paged matmul intern block one thread time: " << duration.count() << " seconds" << std::endl;
}