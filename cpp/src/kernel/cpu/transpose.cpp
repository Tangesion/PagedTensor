#include "kernel/cpu/transpose.h"

namespace paged_tensor::kernel::cpu
{
    void transpose(const float *inp, float *out, const size_t B, const size_t NH, const size_t H, const size_t D)
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

    void transposeMultiThread(const float *inp, float *out, const size_t B, const size_t NH, const size_t H, const size_t D)
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