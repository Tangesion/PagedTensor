#pragma once
#include "kernel/cpu/attention.h"
#include "runtime/tensor.h"

namespace toy::kernel::launch
{
    using UniquePtr = toy::runtime::Tensor::UniquePtr;

    void attentionForward(UniquePtr &out, UniquePtr &query, UniquePtr &key, UniquePtr &value, UniquePtr &interAttn, const bool isPrefill, const cpu::AttentionType attentionType);

} // namespace toy::kernel::launch