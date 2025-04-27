#include "runtime/buffer.h"
#include <cuda_runtime_api.h>
#include "common/cudaUtiles.h"
#include <stdexcept>
// #include "func/threadPool.h"

using namespace paged_tensor::runtime;

MemoryType Buffer::memoryType()
{
    cudaPointerAttributes attributes{};
    // TLLM_CUDA_CHECK(::cudaPointerGetAttributes(&attributes, data));
    switch (attributes.type)
    {
    case cudaMemoryTypeHost:
        return MemoryType::kPINNEDPOOL;
    case cudaMemoryTypeDevice:
        return MemoryType::kGPU;
    case cudaMemoryTypeManaged:
        return MemoryType::kUVM;
    case cudaMemoryTypeUnregistered:
        return MemoryType::kCPU;
    }
    // TLLM_THROW("Unsupported memory type");
    throw std::runtime_error("Unsupported memory type");
}