// (bs, length, head, dim) -> (bs, head, length, dim)
// 固定head, [bs, i, head0, dim] -> [bs, head0, i, dim]

#pragma once
#include <cmath>
#include <omp.h>

#ifndef THREADS_NUM
#define THREADS_NUM 56
#endif

namespace paged_tensor::kernel::cpu
{
    void transpose(const float *inp, float *out, const size_t B, const size_t NH, const size_t H, const size_t D);

    void transposeMultiThread(const float *inp, float *out, const size_t B, const size_t NH, const size_t H, const size_t D);
}