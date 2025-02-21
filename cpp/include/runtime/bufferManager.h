#pragma once

#include "buffer.h"
#include "tensor.h"
#include "common/assert.h"
#include "common/dataType.h"
#include "runtime/llmBuffer.h"

#include <cstring>
#include <memory>

#define BLOCK_SIZE 16
#define BLOCK_NUM 5

namespace toy::runtime
{

    class BufferManager
    {
    public:
        using BufferPtr = toy::runtime::Buffer::UniquePtr;
        using TensorPtr = toy::runtime::Tensor::UniquePtr;

        explicit BufferManager() = default;
        ~BufferManager() = default;

        static auto constexpr KBYTE_TYPE = DataType::kUINT8;
        [[nodiscard]] static TensorPtr cpu(Dims dims, DataType type = KBYTE_TYPE);
        [[nodiscard]] static TensorPtr cpuPaged(Dims dims, DataType type = KBYTE_TYPE, size_t blockNum = BLOCK_NUM, size_t blockSize = BLOCK_SIZE);
        [[nodiscard]] static TensorPtr gpu(Dims dims, DataType type = KBYTE_TYPE);
    };
}