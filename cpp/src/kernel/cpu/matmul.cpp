#include "kernel/cpu/matmul.h"

namespace paged_tensor::kernel::cpu
{

    // inp (B, T, C) weight (OC, C) bias (OC) out (B, T, OC)
    void matmulWeight(float *out, const float *inp, const float *weight, const float *bias, const size_t B, const size_t H, const size_t C, const size_t OC)
    {
        for (size_t b = 0; b < B; b++)
        {
            for (size_t t = 0; t < H; t++)
            {
                float *outBT = out + b * H * OC + t * OC;
                const float *inpBT = inp + b * H * C + t * C;
                for (size_t oc = 0; oc < OC; oc++)
                {
                    const float *weightRow = weight + oc * C;
                    float sum = 0;
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
        const size_t oc)
    {
        float *outBT = out + b * H * OC + h * OC;
        const float *inpBT = inp + b * H * C + h * C;
        const float *weightRow = weight + oc * C;
        float sum = 0;
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

    void matmulWeightThreadPool(
        float *out,
        const float *inp,
        const float *weight,
        const float *bias,
        const size_t B,
        const size_t H,
        const size_t C,
        const size_t OC)
    {
        paged_tensor::func::ThreadPool threadPool(THREADS_NUM);
        for (size_t b = 0; b < B; b++)
        {
            for (size_t h = 0; h < H; h++)
            {
                for (size_t oc = 0; oc < OC; oc++)
                {
                    threadPool.enqueue(matmulWeightPerThreadFunc, out, inp, weight, bias, B, H, C, OC, b, h, oc);
                }
            }
        }
    }

    void matmulWeightMultiThread(float *out, const float *inp, const float *weight, const float *bias, const size_t B, const size_t H, const size_t C, const size_t OC)
    {
#pragma omp parallel for collapse(3) num_threads(THREADS_NUM)
        for (size_t b = 0; b < B; b++)
        {
            for (size_t t = 0; t < H; t++)
            {
                for (size_t oc = 0; oc < OC; oc++)
                {
                    float *outBT = out + b * H * OC + t * OC;
                    const float *inpBT = inp + b * H * C + t * C;
                    const float *weightRow = weight + oc * C;
                    float sum = 0;
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

} // namespace paged_tensor::kernel::cpu