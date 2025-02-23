#pragma once
#include "kernel/cpu/transpose.h"
#include "func/func.h"

namespace paged_tensor::kernel::launch
{
    using UniquePtr = paged_tensor::runtime::Tensor::UniquePtr;

    void transposeOneThread(UniquePtr &out, UniquePtr &inp);

    void transposeMultiThread(UniquePtr &out, UniquePtr &inp);

    void transpose(UniquePtr &out, UniquePtr &inp, const bool isMultiThread = true);

}