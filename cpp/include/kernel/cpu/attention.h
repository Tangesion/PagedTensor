#pragma once
#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <limits>
#include <memory>
#include <cmath>

#ifndef THREADS_NUM
#define THREADS_NUM 56
#endif

namespace paged_tensor::kernel::cpu
{
    enum AttentionType
    {
        kAttentionOneThread,
        kAttentionMultiThread,
    };

    // prefill
    // query key value (B, NH, H, D)
    // out (B, NH, H, D)
    // decode
    // query(B, NH, 1, D)
    // key(B, NH, H, D)
    // value(B, NH, H, D)

    void attentionForwardMultiThread(
        float *out, const float *query, const float *key, const float *value, float *interAttn,
        bool isPrefill,
        const size_t B, const size_t NH, const size_t H, const size_t D);

    void attentionForwardOneThread(
        float *out, const float *query, const float *key, const float *value, float *interAttn,
        bool isPrefill,
        const size_t B, const size_t NH, const size_t H, const size_t D);

} // namespace paged_tensor::kernel::cpu