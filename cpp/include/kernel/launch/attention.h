#pragma once
#include "kernel/cpu/attention.h"
#include "runtime/tensor.h"

namespace inference_frame::kernel::launch
{
    using UniquePtr = inference_frame::runtime::Tensor::UniquePtr;

    void attentionForward(UniquePtr &out, UniquePtr &query, UniquePtr &key, UniquePtr &value, UniquePtr &interAttn, const bool isPrefill, const cpu::AttentionType attentionType);

} // namespace inference_frame::kernel::launch