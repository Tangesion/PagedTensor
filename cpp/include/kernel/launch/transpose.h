#pragma once
#include "kernel/cpu/transpose.h"
#include "func/func.h"

namespace inference_frame::kernel::launch
{
    using UniquePtr = inference_frame::runtime::Tensor::UniquePtr;

    void transposeOneThread(UniquePtr &out, UniquePtr &inp);

    void transposeMultiThread(UniquePtr &out, UniquePtr &inp);

    void transpose(UniquePtr &out, UniquePtr &inp, const bool isMultiThread = true);

}