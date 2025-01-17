#pragma once
#include <cstdlib>
#include <cmath>
#include <omp.h>

#define THREADS_NUM 56

namespace inference_frame::kernel::cpu
{
    void precomputeFreqsCosSinOneThread(float *freqsCosSin, const size_t dim, const size_t maxPos, const float theta = 10000.0)
    {
        // freqsCosSin (maxPos, 2, dim / 2)
        for (size_t i = 0; i < maxPos; i++)
        {
            for (size_t j = 0; j < dim / 2; j++)
            {
                float freq = 1.0f / powf(theta, 2.0f * j / dim);
                freqsCosSin[i * dim + j] = cosf(i * freq);
                freqsCosSin[i * dim + dim / 2 + j] = sinf(i * freq);
            }
        }
    }

    void precomputeFreqsCosSinMultiThread(float *freqsCosSin, const size_t dim, const size_t maxPos, const float theta = 10000.0)
    {
#pragma omp parallel for collapse(2) num_threads(THREADS_NUM)
        for (size_t i = 0; i < maxPos; i++)
        {
            for (size_t j = 0; j < dim / 2; j++)
            {
                float freq = 1.0f / powf(theta, 2.0f * j / dim);
                freqsCosSin[i * dim + j] = cosf(i * freq);
                freqsCosSin[i * dim + dim / 2 + j] = sinf(i * freq);
            }
        }
    }

    void applyRopeOneThread(float *inp, const float *freqsCosSin, const size_t B, const size_t NH, const size_t H, const size_t D, const size_t *pos)
    {
        for (size_t b = 0; b < B; b++)
        {
            for (size_t h = 0; h < H; h++)
            {
                // size_t h = 0; h < H; h++
                for (size_t nh = 0; nh < NH; nh++)
                {
                    float *inpBNH = inp + b * NH * H * D + nh * H * D + h * D;
                    const float *freqsCosSinH = freqsCosSin + pos[h] * D;
                    for (size_t qd = 0; qd < D / 2; qd++)
                    {
                        float rotaCosUp = inpBNH[qd] * freqsCosSinH[qd];
                        float rotaCosDown = inpBNH[qd + D / 2] * freqsCosSinH[qd];

                        float rotaSinUp = -inpBNH[qd + D / 2] * freqsCosSinH[D / 2 + qd];
                        float rotaSinDown = inpBNH[qd] * freqsCosSinH[D / 2 + qd];

                        // 将计算结果存储到 outBNH 中
                        inpBNH[qd] = rotaCosUp + rotaSinUp;
                        inpBNH[qd + D / 2] = rotaCosDown + rotaSinDown;
                    }
                }
            }
        }
    }

    void applyRopeMultiThread(float *inp, const float *freqsCosSin, const size_t B, const size_t NH, const size_t H, const size_t D, const size_t *pos)
    {
#pragma omp parallel for collapse(3) num_threads(THREADS_NUM)
        for (size_t b = 0; b < B; b++)
        {
            for (size_t h = 0; h < H; h++)
            {
                for (size_t nh = 0; nh < NH; nh++)
                {
                    float *inpBNH = inp + b * NH * H * D + nh * H * D + h * D;
                    const float *freqsCosSinH = freqsCosSin + pos[h] * D;
                    for (size_t qd = 0; qd < D / 2; qd++)
                    {
                        float rotaCosUp = inpBNH[qd] * freqsCosSinH[qd];
                        float rotaCosDown = inpBNH[qd + D / 2] * freqsCosSinH[qd];

                        float rotaSinUp = -inpBNH[qd + D / 2] * freqsCosSinH[D / 2 + qd];
                        float rotaSinDown = inpBNH[qd] * freqsCosSinH[D / 2 + qd];

                        // 将计算结果存储到 outBNH 中
                        inpBNH[qd] = rotaCosUp + rotaSinUp;
                        inpBNH[qd + D / 2] = rotaCosDown + rotaSinDown;
                    }
                }
            }
        }
    }
}