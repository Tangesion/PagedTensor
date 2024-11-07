#include "runtime/BufferManager.h"
#include <memory>

namespace inference_frame::runtime
{

    BufferManager::TensorPtr BufferManager::cpu(Dims dims, DataType type)
    {
        return std::make_unique<HostTensor>(dims, type);
    }

}