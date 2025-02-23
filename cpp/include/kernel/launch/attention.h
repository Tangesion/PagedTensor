#pragma once
#include "kernel/cpu/attention.h"
#include "runtime/tensor.h"

namespace paged_tensor::kernel::launch
{
    using UniquePtr = paged_tensor::runtime::Tensor::UniquePtr;

    void attentionForward(UniquePtr &out, UniquePtr &query, UniquePtr &key, UniquePtr &value, UniquePtr &interAttn, const bool isPrefill, const cpu::AttentionType attentionType);

} // namespace paged_tensor::kernel::launch