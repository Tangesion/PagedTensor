#pragma once
#include "kernel/cpu/rmsnorm.h"
#include "runtime/tensor.h"

namespace paged_tensor::kernel::launch
{
    using UniquePtr = paged_tensor::runtime::Tensor::UniquePtr;

    void rmsNorm(UniquePtr &out, UniquePtr &inp, UniquePtr &weight, const bool isMultiThread);
}
