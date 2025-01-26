#pragma once
#include "runtime/tensor.h"
#include "kernel/cpu/matmul.h"

namespace toy::kernel::launch
{
    using UniquePtr = toy::runtime::Tensor::UniquePtr;

    void matmulWeight(UniquePtr &out, UniquePtr &inp, UniquePtr &weight, const UniquePtr &bias, const cpu::MatmulType matmulType);

} // namespace toy::kernel::launch
