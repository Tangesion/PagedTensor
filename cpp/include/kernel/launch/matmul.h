#pragma once
#include "runtime/tensor.h"
#include "kernel/cpu/matmul.h"

namespace paged_tensor::kernel::launch
{
    using UniquePtr = paged_tensor::runtime::Tensor::UniquePtr;

    void matmulWeight(UniquePtr &out, UniquePtr &inp, UniquePtr &weight, const UniquePtr &bias, const cpu::MatmulType matmulType);

} // namespace paged_tensor::kernel::launch
