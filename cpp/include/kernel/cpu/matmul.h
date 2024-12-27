#pragma once
#include <iostream>
// #include <omp.h>
#include <cstdlib>
#include <memory>
#include "func/threadPool.h"
#include "func/func.h"
#include "runtime/tensor.h"

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

    using namespace inference_frame::runtime;
    void matmulWeightLaunch(Tensor::SharedPtr out, Tensor::SharedPtr inp, Tensor::SharedPtr weight, Tensor::SharedPtr bias, MatmulType matmulType)
    {

        DataType dataTypeInp = inp->getDataType();
        DataType dataTypeWeight = weight->getDataType();
        DataType dataTypeOut = out->getDataType();
        try
        {
            CHECK_WITH_INFO(dataTypeInp == dataTypeWeight, "Data type of input and weight must be the same");
            CHECK_WITH_INFO(dataTypeOut == dataTypeInp, "Data type of output and input must be the same");
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            std::exit(EXIT_FAILURE);
        }

        int64_t B = inp->getShape().d[0];
        int64_t H = inp->getShape().d[1];
        int64_t C = inp->getShape().d[2];
        int64_t OC = weight->getShape().d[0];

        auto *outData = inference_frame::func::getData<Tensor::DataType::kFLOAT>(out);
        auto *inpData = inference_frame::func::getData<Tensor::DataType::kFLOAT>(inp);
        auto *weightData = inference_frame::func::getData<Tensor::DataType::kFLOAT>(weight);
        auto *biasData = bias == nullptr ? nullptr : inference_frame::func::getData<Tensor::DataType::kFLOAT>(bias);

        switch (matmulType)
        {
        case MatmulType::kMatmulOneThread:
            matmulWeight(outData, inpData, weightData, biasData, B, H, C, OC);
            break;
        case MatmulType::KMatmulMultiThread:
            matmulWeightMultiThread(outData, inpData, weightData, biasData, B, H, C, OC);
            break;
        case MatmulType::kMatmulThreadPool:
            matmulWeightThreadPool(outData, inpData, weightData, biasData, B, H, C, OC);
            break;
        }

        matmulWeight(outData, inpData, weightData, biasData, B, H, C, OC);

        // matmul<float>(static_cast<float *>(out->data()), static_cast<float *>(inp->data()), static_cast<float *>(weight->data()), nullptr, 1, 2, 3, 4);
    }

} // namespace inference_frame::kernel::cpu