#pragma once
#include <cstdlib>
#include <cmath>
#include <omp.h>

#ifndef THREADS_NUM
#define THREADS_NUM 56
#endif

namespace inference_frame::kernel::cpu
{
    void rmsNormOneThread(float *out, const float *inp, const float *weight, const size_t B, const size_t H, const size_t C);

    void rmsNormMultiThread(float *out, const float *inp, const float *weight, const size_t B, const size_t H, const size_t C);
}