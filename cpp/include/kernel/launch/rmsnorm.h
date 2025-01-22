#pragma once
#include "kernel/cpu/rmsnorm.h"
#include "runtime/tensor.h"

namespace inference_frame::kernel::launch
{
    using UniquePtr = inference_frame::runtime::Tensor::UniquePtr;

    void rmsNorm(UniquePtr &out, UniquePtr &inp, UniquePtr &weight, const bool isMultiThread);
}
