// (bs, length, head, dim) -> (bs, head, length, dim)
// 固定head, [bs, i, head0, dim] -> [bs, head0, i, dim]

#pragma once
#include <cmath>
#include <omp.h>

#define THREADS_NUM 56

namespace inference_frame::kernel::cpu
{
    template <typename T>
    void transpose(const T *inp, T *out, const size_t B, const size_t NH, const size_t H, const size_t D)
    {
        for (size_t b = 0; b < B; b++)
        {
            for (size_t nh = 0; nh < NH; nh++)
            {
                for (size_t h = 0; h < H; h++)
                {
                    for (size_t d = 0; d < D; d++)
                    {
                        out[b * NH * H * D + nh * H * D + h * D + d] = inp[b * NH * H * D + h * NH * D + nh * D + d];
                    }
                }
            }
        }
    }

    template <typename T>
    void transposeMultiThread(const T *inp, T *out, const size_t B, const size_t NH, const size_t H, const size_t D)
    {
#pragma omp parallel for collapse(4) num_threads(THREADS_NUM)
        for (size_t b = 0; b < B; b++)
        {
            for (size_t nh = 0; nh < NH; nh++)
            {
                for (size_t h = 0; h < H; h++)
                {
                    for (size_t d = 0; d < D; d++)
                    {
                        out[b * NH * H * D + nh * H * D + h * D + d] = inp[b * NH * H * D + h * NH * D + nh * D + d];
                    }
                }
            }
        }
    }
}