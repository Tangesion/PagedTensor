#pragma once
// #include <omp.h>
#include <cstdlib>
#include <memory>
#include "func/threadPool.h"

#define THREADS_NUM 28

namespace inference_frame::kernel::cpu
{

    enum class MatmulType
    {
        kMatmulOneThread,
        KMatmulMultiThread,
        kMatmulThreadPool,
    };

    // inp (B, T, C) weight (OC, C) bias (OC) out (B, T, OC)
    template <typename T>
    void matmulWeight(T *out, const T *inp, const T *weight, const T *bias, const size_t B, const size_t H, const size_t C, const size_t OC)
    {
        for (size_t b = 0; b < B; b++)
        {
            for (size_t t = 0; t < H; t++)
            {
                T *outBT = out + b * H * OC + t * OC;
                const T *inpBT = inp + b * H * C + t * C;
                for (size_t oc = 0; oc < OC; oc++)
                {
                    const T *weightRow = weight + oc * C;
                    T sum = 0;
                    for (size_t c = 0; c < C; c++)
                    {
                        sum += inpBT[c] * weightRow[c];
                    }
                    if (bias != nullptr)
                    {
                        outBT[oc] = sum + bias[oc];
                    }
                    else
                    {
                        outBT[oc] = sum;
                    }
                }
            }
        }
    }

    template <typename T>
    void matmulWeightPerThreadFunc(
        T *out,
        const T *inp,
        const T *weight,
        const T *bias,
        const size_t B,
        const size_t H,
        const size_t C,
        const size_t OC,
        const size_t b,
        const size_t h,
        const size_t oc)
    {
        T *outBT = out + b * H * OC + h * OC;
        const T *inpBT = inp + b * H * C + h * C;
        const T *weightRow = weight + oc * C;
        T sum = 0;
        for (size_t c = 0; c < C; c++)
        {
            sum += inpBT[c] * weightRow[c];
        }
        if (bias != nullptr)
        {
            outBT[oc] = sum + bias[oc];
        }
        else
        {
            outBT[oc] = sum;
        }
    }

    template <typename T>
    void matmulWeightThreadPool(
        T *out,
        const T *inp,
        const T *weight,
        const T *bias,
        const size_t B,
        const size_t H,
        const size_t C,
        const size_t OC)
    {
        inference_frame::func::ThreadPool threadPool(THREADS_NUM);
        for (size_t b = 0; b < B; b++)
        {
            for (size_t h = 0; h < H; h++)
            {
                for (size_t oc = 0; oc < OC; oc++)
                {
                    threadPool.enqueue(matmulWeightPerThreadFunc<T>, out, inp, weight, bias, B, H, C, OC, b, h, oc);
                }
            }
        }
    }
    template <typename T>
    void matmulWeightMultiThread(T *out, const T *inp, const T *weight, const T *bias, const size_t B, const size_t H, const size_t C, const size_t OC)
    {
#pragma omp parallel for collapse(3) num_threads(THREADS_NUM)
        for (size_t b = 0; b < B; b++)
        {
            for (size_t t = 0; t < H; t++)
            {
                for (size_t oc = 0; oc < OC; oc++)
                {
                    T *outBT = out + b * H * OC + t * OC;
                    const T *inpBT = inp + b * H * C + t * C;
                    const T *weightRow = weight + oc * C;
                    T sum = 0;
                    for (size_t c = 0; c < C; c++)
                    {
                        sum += inpBT[c] * weightRow[c];
                    }
                    if (bias != nullptr)
                    {
                        outBT[oc] = sum + bias[oc];
                    }
                    else
                    {
                        outBT[oc] = sum;
                    }
                }
            }
        }
    }

    // matmul<float>(static_cast<float *>(out->data()), static_cast<float *>(inp->data()), static_cast<float *>(weight->data()), nullptr, 1, 2, 3, 4);

} // namespace inference_frame::kernel::cpu