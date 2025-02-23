#pragma once
#include <iostream>
#include "kernel/cpu/ffn.h"
#include "runtime/tensor.h"
#include "func/func.h"

namespace paged_tensor::kernel::launch
{
    using UniquePtr = paged_tensor::runtime::Tensor::UniquePtr;

    void ffnForwardOneThread(UniquePtr &out, UniquePtr &inp, UniquePtr &gateProj, UniquePtr &upProj, UniquePtr &downProj);

    void ffnForwardMultiThreads(UniquePtr &out, UniquePtr &inp, UniquePtr &gateProj, UniquePtr &upProj, UniquePtr &downProj);

    void ffnForward(UniquePtr &out, UniquePtr &inp, UniquePtr &gateProj, UniquePtr &upProj, UniquePtr &downProj, const bool isMultiThread = true);
}