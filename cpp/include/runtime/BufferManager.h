#pragma once

#include "Buffer.h"
#include "Tensor.h"
#include "common/assert.h"
#include "common/DataType.h"
#include "runtime/llmBuffer.h"

#include <cstring>
#include <memory>

namespace inference_frame::runtime
{
    class BufferManager
    {
    public:
        using BufferPtr = inference_frame::runtime::Buffer::UniquePtr;
        using TensorPtr = inference_frame::runtime::Tensor::UniquePtr;

        explicit BufferManager() = default;
        ~BufferManager() = default;

        static auto constexpr KBYTE_TYPE = DataType::kUINT8;
        [[nodiscard]] static TensorPtr cpu(Dims dims, DataType type = KBYTE_TYPE);
    };
}