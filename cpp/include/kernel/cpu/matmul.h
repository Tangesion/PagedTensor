#pragma once
// #include <omp.h>
#include <cstdlib>
#include <memory>
#include "func/threadPool.h"
#include "common/dataPtr.h"
#ifndef THREADS_NUM
#define THREADS_NUM 56
#endif

using DataPtr = paged_tensor::common::DataPtr;

namespace paged_tensor::kernel::cpu
{
    // using namespace paged_tensor::common;

    enum class MatmulType
    {
        kMatmulOneThread,
        kMatmulMultiThread,
        kMatmulThreadPool,
    };

    // inp (B, T, C) weight (OC, C) bias (OC) out (B, T, OC)
    void matmulWeight(float *out, const float *inp, const float *weight, const float *bias, const size_t B, const size_t H, const size_t C, const size_t OC);

    void matmulWeightPaged(DataPtr out, DataPtr inp, const float *weight, const float *bias, const size_t B, const size_t H, const size_t C, const size_t OC);

    void matmulWeightPagedMultiThread(DataPtr out, DataPtr inp, const float *weight, const float *bias, const size_t B, const size_t H, const size_t C, const size_t OC);

    void matmulWeightPerThreadFunc(
        float *out,
        const float *inp,
        const float *weight,
        const float *bias,
        const size_t B,
        const size_t H,
        const size_t C,
        const size_t OC,
        const size_t b,
        const size_t h,
        const size_t oc);

    void matmulWeightThreadPool(
        float *out,
        const float *inp,
        const float *weight,
        const float *bias,
        const size_t B,
        const size_t H,
        const size_t C,
        const size_t OC);

    void matmulWeightMultiThread(float *out, const float *inp, const float *weight, const float *bias, const size_t B, const size_t H, const size_t C, const size_t OC);

    // matmul<float>(static_cast<float *>(out->data()), static_cast<float *>(inp->data()), static_cast<float *>(weight->data()), nullptr, 1, 2, 3, 4);

} // namespace paged_tensor::kernel::cpu