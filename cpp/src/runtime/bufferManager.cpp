#include "runtime/bufferManager.h"
#include <memory>

namespace toy::runtime
{

    BufferManager::TensorPtr BufferManager::cpu(Dims dims, DataType type)
    {
        return std::make_unique<HostTensor>(dims, type);
    }

    BufferManager::TensorPtr BufferManager::cpuPaged(Dims dims, DataType type, size_t blockNum, size_t blockSize)
    {
        BlockManager::getInstance().initialize(blockNum, blockSize, type);
        return std::make_unique<HostPagedTensor>(dims, type, true);
    }

    BufferManager::TensorPtr BufferManager::gpu(Dims dims, DataType type)
    {
        // return std::make_unique<>
    }

}