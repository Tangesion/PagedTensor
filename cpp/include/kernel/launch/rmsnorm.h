#pragma once
#include "kernel/cpu/rmsnorm.h"
#include "runtime/tensor.h"

namespace toy::kernel::launch
{
    using UniquePtr = toy::runtime::Tensor::UniquePtr;

    void rmsNorm(UniquePtr &out, UniquePtr &inp, UniquePtr &weight, const bool isMultiThread);
}
