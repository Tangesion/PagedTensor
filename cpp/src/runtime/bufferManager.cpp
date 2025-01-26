#include "runtime/bufferManager.h"
#include <memory>

namespace toy::runtime
{

    BufferManager::TensorPtr BufferManager::cpu(Dims dims, DataType type)
    {
        return std::make_unique<HostTensor>(dims, type);
    }

    BufferManager::TensorPtr BufferManager::gpu(Dims dims, DataType type)
    {
        // return std::make_unique<>
    }

}