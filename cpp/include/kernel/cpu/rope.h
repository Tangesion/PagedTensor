#pragma once
#include <cstdlib>
#include <cmath>
#include <omp.h>

#define THREADS_NUM 56

namespace paged_tensor::kernel::cpu
{
    void precomputeFreqsCosSinOneThread(float *freqsCosSin, const size_t dim, const size_t maxPos, const float theta = 10000.0);

    void precomputeFreqsCosSinMultiThread(float *freqsCosSin, const size_t dim, const size_t maxPos, const float theta = 10000.0);

    void applyRopeOneThread(float *inp, const float *freqsCosSin, const size_t B, const size_t NH, const size_t H, const size_t D, const size_t *pos);

    void applyRopeMultiThread(float *inp, const float *freqsCosSin, const size_t B, const size_t NH, const size_t H, const size_t D, const size_t *pos);
}