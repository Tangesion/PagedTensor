#pragma once

#include "buffer.h"
#include "tensor.h"
#include "common/assert.h"
#include "common/dataType.h"
#include "runtime/llmBuffer.h"

#include <cstring>
#include <memory>

#define BLOCK_SIZE 4096
#define BLOCK_NUM 1024

namespace paged_tensor::runtime
{

    class BufferManager
    {
    public:
        using BufferPtr = paged_tensor::runtime::Buffer::UniquePtr;
        using TensorPtr = paged_tensor::runtime::Tensor::UniquePtr;

        explicit BufferManager() = default;
        ~BufferManager() = default;

        static auto constexpr KBYTE_TYPE = DataType::kUINT8;
        [[nodiscard]] static TensorPtr cpu(Dims dims, DataType type = KBYTE_TYPE);
        [[nodiscard]] static TensorPtr cpuPaged(Dims dims, DataType type = KBYTE_TYPE, size_t blockNum = BLOCK_NUM, size_t blockSize = BLOCK_SIZE);
        [[nodiscard]] static TensorPtr gpu(Dims dims, DataType type = KBYTE_TYPE);
    };
}