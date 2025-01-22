#include "kernel/cpu/rmsnorm.h"

namespace inference_frame::kernel::cpu
{

    void rmsNormOneThread(float *out, const float *inp, const float *weight, const size_t B, const size_t H, const size_t C)
    {
        for (size_t b = 0; b < B; b++)
        {
            for (size_t h = 0; h < H; h++)
            {
                float ss = 0;
                const float *inpBT = inp + b * H * C + h * C;
                for (size_t c = 0; c < C; c++)
                {
                    ss += inpBT[c] * inpBT[c];
                }
                ss /= C;
                ss += 1e-6;
                ss = 1 / std::sqrt(ss);
                float *outBT = out + b * H * C + h * C;
                for (size_t c = 0; c < C; c++)
                {
                    outBT[c] = weight[c] * inpBT[c] * ss;
                }
            }
        }
    }

    void rmsNormMultiThread(float *out, const float *inp, const float *weight, const size_t B, const size_t H, const size_t C)
    {
#pragma omp parallel for collapse(2) num_threads(THREADS_NUM)
        for (size_t b = 0; b < B; b++)
        {
            for (size_t h = 0; h < H; h++)
            {
                float ss = 0;
                const float *inpBT = inp + b * H * C + h * C;
                for (size_t c = 0; c < C; c++)
                {
                    ss += inpBT[c] * inpBT[c];
                }
                ss /= C;
                ss += 1e-6;
                ss = 1 / std::sqrt(ss);
                float *outBT = out + b * H * C + h * C;
                for (size_t c = 0; c < C; c++)
                {
                    outBT[c] = weight[c] * inpBT[c] * ss;
                }
            }
        }
    }
}