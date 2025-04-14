#include "kernel/cpu/matmul.h"
#include <iostream>

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
    void matmulWeightPaged(DataPtr out, DataPtr inp, const float *weight, const float *bias, const size_t B, const size_t H, const size_t C, const size_t OC)
    {
        for (size_t b = 0; b < B; b++)
        {
            for (size_t t = 0; t < H; t++)
            {
                DataPtr outBT = out + b * H * OC + t * OC;
                DataPtr inpBT = inp + b * H * C + t * C;

                for (size_t oc = 0; oc < OC; oc++)
                {
                    const float *weightRow = weight + oc * C;
                    float sum = 0;
                    for (size_t c = 0; c < C; c++)
                    {
                        sum += *(inpBT[c].data<float>()) * weightRow[c];
                    }
                    if (bias != nullptr)
                    {
                        *(outBT[oc].data<float>()) = sum + bias[oc];
                    }
                    else
                    {
                        *(outBT[oc].data<float>()) = sum;
                    }
                }
            }
        }
    }

    void matmulWeightPagedInternBlock(DataPtr out, DataPtr inp, const float *weight, const float *bias, const size_t B, const size_t H, const size_t C, const size_t OC)
    {
        for (size_t b = 0; b < B; b++)
        {
            for (size_t t = 0; t < H; t++)
            {
                DataPtr outBT = out + b * H * OC + t * OC;
                DataPtr inpBT = inp + b * H * C + t * C;

                for (size_t oc = 0; oc < OC; oc++)
                {
                    const float *weightRow = weight + oc * C;
                    float sum = 0;
                    for (size_t c = 0; c < C; c++)
                    {
                        sum += inpBT.data<float>(c) * weightRow[c];
                    }
                    if (bias != nullptr)
                    {
                        outBT.data<float>(oc) = sum + bias[oc];
                    }
                    else
                    {
                        outBT.data<float>(oc) = sum;
                    }
                }
            }
        }
    }

    void matmulWeightPagedBlock(DataPtr out, DataPtr inp, const float *weight, const float *bias, const size_t B, const size_t H, const size_t C, const size_t OC)
    {
        for (size_t b = 0; b < B; b++)
        {
            for (size_t t = 0; t < H; t++)
            {
                for (size_t oc = 0; oc < OC; oc++)
                {
                    DataPtr outBT = out + b * H * OC + t * OC;
                    DataPtr inpBT = inp + b * H * C + t * C;
                    const float *weightRow = weight + oc * C;
                    float sum = 0;

                    // 1 2 3 4 |  5 6 7 8 |  9 10 11 12
                    //     |                    |
                    // hos                  tos
                    // 分三个块

                    size_t headOffset = inpBT.getBlockOffset();
                    size_t blockSize = inpBT.getBlockSize();
                    size_t headBlockLength = blockSize - headOffset;
                    size_t dataLength = C;

                    DataPtr inpBTX = inpBT[0];
                    DataPtr inpBTXEnd = inpBT[dataLength];

                    size_t blockNum = inpBTXEnd.getBlockIdx() - inpBTX.getBlockIdx() + 1;

                    size_t tailOffset = inpBTXEnd.getBlockOffset();

                    // std::cout << "headOffset " << headOffset << std::endl;
                    // std::cout << "blockSize " << blockSize << std::endl;
                    // std::cout << "headBlockLength " << headBlockLength << std::endl;
                    // std::cout << "dataLength" << dataLength << std::endl;
                    // std::cout << "tailOffset" << tailOffset << std::endl;
                    // std::cout << "blockNum" << blockNum << std::endl;
                    // std::cout << "t" << t << std::endl;
                    // std::cout << "oc" << oc << std::endl;
                    // std::cout << std::endl;
                    int c = 0;
                    if (blockNum == 1)
                    {
                        for (size_t cc = 0; cc < dataLength; cc++, c++)
                        {
                            sum += inpBTX.data<float>()[cc] * weightRow[c];
                        }
                    }
                    else
                    {
                        for (size_t bx = 0; bx < blockNum; bx++)
                        {
                            // std::cout << inpBTX.getBlockIdx() << std::endl;
                            //  std::cout << bx << std::endl;
                            if (bx == 0)
                            {
                                for (size_t cc = 0; cc < headBlockLength; cc++, c++)
                                {
                                    sum += inpBTX.data<float>()[cc] * weightRow[c];
                                }
                                inpBTX = inpBTX + (headBlockLength);
                            }
                            else if (bx == blockNum - 1)
                            {
                                for (size_t cc = 0; cc < tailOffset; cc++, c++)
                                {
                                    sum += inpBTX.data<float>()[cc] * weightRow[c];
                                }
                            }
                            else
                            {

                                for (size_t cc = 0; cc < blockSize; cc++, c++)
                                {
                                    // std::cout << cc << " ";
                                    sum += inpBTX.data<float>()[cc] * weightRow[c];
                                }
                                // std::cout << std::endl;
                                inpBTX = inpBTX + blockSize;
                            }
                        }
                    }

                    if (bias != nullptr)
                    {
                        *(outBT[oc].data<float>()) = sum + bias[oc];
                    }
                    else
                    {
                        *(outBT[oc].data<float>()) = sum;
                    }
                }
            }
        }
    }

    void matmulWeightPagedBlockMultiThread(DataPtr out, DataPtr inp, const float *weight, const float *bias, const size_t B, const size_t H, const size_t C, const size_t OC)
    {
#pragma omp parallel for collapse(3) schedule(dynamic) num_threads(THREADS_NUM)
        for (size_t b = 0; b < B; b++)
        {
            for (size_t t = 0; t < H; t++)
            {
                for (size_t oc = 0; oc < OC; oc++)
                {
                    DataPtr outBT = out + b * H * OC + t * OC;
                    DataPtr inpBT = inp + b * H * C + t * C;
                    const float *weightRow = weight + oc * C;
                    float sum = 0;

                    // 1 2 3 4 |  5 6 7 8 |  9 10 11 12
                    //     |                    |
                    // hos                  tos
                    // 分三个块

                    size_t headOffset = inpBT.getBlockOffset();
                    size_t blockSize = inpBT.getBlockSize();
                    size_t headBlockLength = blockSize - headOffset;
                    size_t dataLength = C;

                    DataPtr inpBTX = inpBT[0];
                    DataPtr inpBTXEnd = inpBT[dataLength];

                    size_t blockNum = inpBTXEnd.getBlockIdx() - inpBTX.getBlockIdx() + 1;

                    size_t tailOffset = inpBTXEnd.getBlockOffset();

                    // std::cout << "headOffset " << headOffset << std::endl;
                    // std::cout << "blockSize " << blockSize << std::endl;
                    // std::cout << "headBlockLength " << headBlockLength << std::endl;
                    // std::cout << "dataLength" << dataLength << std::endl;
                    // std::cout << "tailOffset" << tailOffset << std::endl;
                    // std::cout << "blockNum" << blockNum << std::endl;
                    // std::cout << "t" << t << std::endl;
                    // std::cout << "oc" << oc << std::endl;
                    // std::cout << std::endl;
                    int c = 0;
                    if (blockNum == 1)
                    {
                        for (size_t cc = 0; cc < dataLength; cc++, c++)
                        {
                            sum += inpBTX.data<float>()[cc] * weightRow[c];
                        }
                    }
                    else
                    {
                        for (size_t bx = 0; bx < blockNum; bx++)
                        {
                            // std::cout << inpBTX.getBlockIdx() << std::endl;
                            //  std::cout << bx << std::endl;
                            if (bx == 0)
                            {
                                for (size_t cc = 0; cc < headBlockLength; cc++, c++)
                                {
                                    sum += inpBTX.data<float>()[cc] * weightRow[c];
                                }
                                inpBTX = inpBTX + (headBlockLength);
                            }
                            else if (bx == blockNum - 1)
                            {
                                for (size_t cc = 0; cc < tailOffset; cc++, c++)
                                {
                                    sum += inpBTX.data<float>()[cc] * weightRow[c];
                                }
                            }
                            else
                            {

                                for (size_t cc = 0; cc < blockSize; cc++, c++)
                                {
                                    // std::cout << cc << " ";
                                    sum += inpBTX.data<float>()[cc] * weightRow[c];
                                }
                                // std::cout << std::endl;
                                inpBTX = inpBTX + blockSize;
                            }
                        }
                    }

                    if (bias != nullptr)
                    {
                        *(outBT[oc].data<float>()) = sum + bias[oc];
                    }
                    else
                    {
                        *(outBT[oc].data<float>()) = sum;
                    }
                }
            }
        }
    }

    void matmulWeightPagedMultiThread(DataPtr out, DataPtr inp, const float *weight, const float *bias, const size_t B, const size_t H, const size_t C, const size_t OC)
    {
#pragma omp parallel for collapse(3) num_threads(THREADS_NUM)
        for (size_t b = 0; b < B; b++)
        {
            for (size_t t = 0; t < H; t++)
            {

                for (size_t oc = 0; oc < OC; oc++)
                {
                    DataPtr outBT = out + b * H * OC + t * OC;
                    DataPtr inpBT = inp + b * H * C + t * C;
                    const float *weightRow = weight + oc * C;
                    float sum = 0;
                    for (size_t c = 0; c < C; c++)
                    {
                        sum += *(inpBT[c].data<float>()) * weightRow[c];
                    }
                    if (bias != nullptr)
                    {
                        *(outBT[oc].data<float>()) = sum + bias[oc];
                    }
                    else
                    {
                        *(outBT[oc].data<float>()) = sum;
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