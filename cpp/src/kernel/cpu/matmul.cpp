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
                    DataPtr inpBTBlock = inpBT[0];
                    size_t offset = inpBTBlock.getBlockOffset();
                    size_t blockSize = inpBTBlock.getBlockSize();
                    size_t currentBlockLength = blockSize - offset;
                    for (size_t c = 0; c < C; c++, offset++)
                    {
                        /*
                        编译器考虑到inpBTBlock可能会在for循环中发生变化
                        所以不能直接通过 movss   (%rdx,%rcx,4),%xmm0   从 inpBTX.data<float>()[cc] 加载浮点值，
                        因为这里的rdx是基地址，是个定值，而inpBTBlock发生了变化
                        所以原来是
                        movss   (%rdx,%rcx,4),%xmm0  // 从 inpBTX.data<float>()[cc] 加载浮点值
                        mulss   (%r15,%rcx,4),%xmm0  // 乘以 weightRow[c]
                        addss   %xmm0,%xmm1          // 将结果累加到 sum
                        仅需三条指令

                        而现在需要
                        lea     0x0(,%rdi,4),%rsi   // 计算 offset * 4 的字节偏移量，存储到 %rsi
                        add     (%r11,%rax,8),%rsi  // 加上块的基地址，得到最终地址，存储到 %rsi
                        movss   (%rsi,%rdx,1),%xmm0 // 从最终地址加载浮点值到 %xmm0
                        mulss   -0x4(%rcx),%xmm0    // 将浮点值乘以 weightRow[c]
                        addss   %xmm0,%xmm1         // 将乘积累加到 sum
                        五条指令

                        rdi : offset的值
                        r11 : 块的基地址
                        rax : 块索引
                        rsi : 内存地址
                        */
                        if (offset >= blockSize)
                        {
                            inpBTBlock = inpBTBlock + currentBlockLength;
                            offset = 0;
                            currentBlockLength = blockSize - offset;
                        }
                        sum += inpBTBlock.data<float>()[offset] * weightRow[c];
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
    inline void cacheBlocks(DataPtr dataPtr, float *cache, size_t blockSize, size_t C) __attribute__((always_inline));
    inline void cacheBlocks(DataPtr dataPtr, float *cache, size_t blockSize, size_t C)
    {
        size_t c = 0;
        DataPtr currentBlock = dataPtr[0];
        size_t headOffset = dataPtr.getBlockOffset();
        size_t blockSizeInp = dataPtr.getBlockSize();
        size_t headBlockLength = blockSize - headOffset;
        DataPtr endBlock = dataPtr[C];
        size_t blockNum = endBlock.getBlockIdx() - currentBlock.getBlockIdx() + 1;
        size_t tailOffset = currentBlock.getBlockOffset();
        for (size_t bx = 0; bx < blockNum; bx++)
        {
            if (bx == 0)
            {
                for (size_t cc = 0; cc < headBlockLength; cc++, c++)
                {
                    cache[c] = currentBlock.data<float>()[cc];
                }
                currentBlock = currentBlock + headBlockLength;
            }
            else if (bx == blockNum - 1)
            {
                for (size_t cc = 0; cc < tailOffset; cc++, c++)
                {
                    cache[c] = currentBlock.data<float>()[cc];
                }
            }
            else
            {
                for (size_t cc = 0; cc < blockSize; cc++, c++)
                {
                    cache[c] = currentBlock.data<float>()[cc];
                }
                currentBlock = currentBlock + blockSize;
            }
        }
    }

    //    void matmulWeightBothPagedBlock(DataPtr out, DataPtr inp, DataPtr weight, const float *bias, const size_t B, const size_t H, const size_t C, const size_t OC)
    //    {
    // #pragma omp parallel for collapse(3) schedule(dynamic) num_threads(THREADS_NUM)
    //        for (size_t b = 0; b < B; b++)
    //        {
    //            for (size_t t = 0; t < H; t++)
    //            {
    //                for (size_t oc = 0; oc < OC; oc++)
    //                {
    //                    DataPtr outBT = out + b * H * OC + t * OC;
    //                    DataPtr inpBT = inp + b * H * C + t * C;
    //                    DataPtr weightRow = weight + oc * C;
    //                    float sum = 0;
    //
    //                    // 1 2 3 4 |  5 6 7 8 |  9 10 11 12
    //                    //     |                    |
    //                    // hos                  tos
    //                    // 分三个块
    //
    //                    size_t blockSize = inpBT.getBlockSize();
    //
    //                    float inpCache[C];
    //                    float weightCache[C];
    //
    //                    cacheBlocks(inpBT, inpCache, blockSize, C);
    //                    cacheBlocks(weightRow, weightCache, blockSize, C);
    //
    //                    for (size_t c = 0; c < C; c++)
    //                    {
    //                        sum += weightCache[c] * inpCache[c];
    //                    }
    //
    //                    if (bias != nullptr)
    //                    {
    //                        *(outBT[oc].data<float>()) = sum + bias[oc];
    //                    }
    //                    else
    //                    {
    //                        *(outBT[oc].data<float>()) = sum;
    //                    }
    //                }
    //            }
    //        }
    //    }

    void matmulWeightBothPagedBlock(DataPtr out, DataPtr inp, DataPtr weight, const float *bias, const size_t B, const size_t H, const size_t C, const size_t OC)
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
                    DataPtr weightRow = weight + oc * C;
                    float sum = 0;

                    // Input block information
                    size_t headOffsetInp = inpBT.getBlockOffset();
                    size_t blockSizeInp = inpBT.getBlockSize();
                    size_t headBlockLengthInp = blockSizeInp - headOffsetInp;
                    DataPtr inpBTX = inpBT[0];
                    DataPtr inpBTXEnd = inpBT[C];
                    size_t blockNumInp = inpBTXEnd.getBlockIdx() - inpBTX.getBlockIdx() + 1;
                    size_t tailOffsetInp = inpBTXEnd.getBlockOffset();

                    // Weight block information
                    size_t headOffsetWeight = weightRow.getBlockOffset();
                    size_t blockSizeWeight = weightRow.getBlockSize();
                    size_t headBlockLengthWeight = blockSizeWeight - headOffsetWeight;
                    DataPtr weightRowX = weightRow[0];
                    DataPtr weightRowEnd = weightRow[C];
                    size_t blockNumWeight = weightRowEnd.getBlockIdx() - weightRowX.getBlockIdx() + 1;
                    size_t tailOffsetWeight = weightRowEnd.getBlockOffset();

                    size_t c = 0;
                    size_t inpBlockIdx = 0, weightBlockIdx = 0;

                    size_t inpBlockLength = 0;
                    size_t weightBlockLength = 0;

                    while (c < C)
                    {
                        if (inpBlockLength == 0)
                        {
                            inpBlockLength = (blockNumInp == 1)                 ? C
                                             : (inpBlockIdx == 0)               ? headBlockLengthInp
                                             : (inpBlockIdx == blockNumInp - 1) ? tailOffsetInp
                                                                                : blockSizeInp;
                        }

                        if (weightBlockLength == 0)
                        {
                            weightBlockLength = (blockNumWeight == 1)                    ? C
                                                : (weightBlockIdx == 0)                  ? headBlockLengthWeight
                                                : (weightBlockIdx == blockNumWeight - 1) ? tailOffsetWeight
                                                                                         : blockSizeWeight;
                        }

                        size_t processLength = std::min(inpBlockLength, weightBlockLength);

                        for (size_t cc = 0; cc < processLength; cc++, c++)
                        {
                            sum += inpBTX.data<float>()[cc] * weightRowX.data<float>()[cc];
                        }

                        // Update input block
                        inpBlockLength -= processLength;
                        if (inpBlockLength == 0 && inpBlockIdx < blockNumInp - 1)
                        {
                            inpBTX = inpBTX + inpBlockLength;
                            inpBlockIdx++;
                        }

                        // Update weight block
                        weightBlockLength -= processLength;
                        if (weightBlockLength == 0 && weightBlockIdx < blockNumWeight - 1)
                        {
                            weightRowX = weightRowX + weightBlockLength;
                            weightBlockIdx++;
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