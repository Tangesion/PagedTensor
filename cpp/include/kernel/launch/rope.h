#pragma once
#include "kernel/cpu/rope.h"
#include "runtime/tensor.h"

namespace toy::kernel::launch
{
    using UniquePtr = toy::runtime::Tensor::UniquePtr;

    void precomputeFreqsCosSin(UniquePtr &freqsCosSin, const size_t dim, const size_t maxPos, const float theta = 10000.0, const bool isMultiThread = true);

    void applyRope(UniquePtr &inp, UniquePtr &freqsCosSin, UniquePtr &pos, const bool isMultiThread = true);
}