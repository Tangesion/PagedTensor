#pragma once
#include "runtime/tensor.h"
#include "kernel/cpu/matmul.h"

namespace inference_frame::kernel::launch
{
    using UniquePtr = inference_frame::runtime::Tensor::UniquePtr;

    void matmulWeight(UniquePtr &out, UniquePtr &inp, UniquePtr &weight, const UniquePtr &bias, const cpu::MatmulType matmulType);

} // namespace inference_frame::kernel::launch
