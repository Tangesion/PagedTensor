#pragma once
#include <cmath>
#include <iostream>
#include "matmul.h"

// down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
namespace inference_frame::kernel::cpu
{
    template <typename T>
    void ffnForwardOneThread(
        T *out,
        const T *inp,
        T *outInterGate,
        T *outInterUp,
        const T *gateProj,
        const T *upProj,
        const T *downProj,
        const size_t B,
        const size_t H,
        const size_t C,
        const size_t interSize)
    {
        // gate_proj(x)
        matmulWeight(
            outInterGate,
            inp,
            gateProj,
            static_cast<const T *>(nullptr),
            B,
            H,
            C,
            interSize);

        // std::cout << "weight1" << std::endl;
        //  up_proj(x)
        matmulWeight(
            outInterUp,
            inp,
            upProj,
            static_cast<const T *>(nullptr),
            B,
            H,
            C,
            interSize);
        // self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        for (size_t b = 0; b < B; b++)
        {
            for (size_t h = 0; h < H; h++)
            {
                T *outInterGateBH = outInterGate + b * H * interSize + h * interSize;
                T *outInterUpBH = outInterUp + b * H * interSize + h * interSize;
                for (size_t i = 0; i < interSize; i++)
                {
                    T value = outInterGateBH[i];
                    value *= 1 / (1 + std::exp(-value));
                    outInterGateBH[i] = value * outInterUpBH[i];
                }
            }
        }
        // down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        matmulWeight(
            out,
            outInterGate,
            downProj,
            static_cast<const T *>(nullptr),
            B,
            H,
            interSize,
            C);
    }

    template <typename T>
    void ffnForwardMultiThread(
        T *out,
        const T *inp,
        T *outInterGate,
        T *outInterUp,
        const T *gateProj,
        const T *upProj,
        const T *downProj,
        const size_t B,
        const size_t H,
        const size_t C,
        const size_t interSize)
    {
        matmulWeightMultiThread(
            outInterGate,
            inp,
            gateProj,
            static_cast<const T *>(nullptr),
            B,
            H,
            C,
            interSize);
        matmulWeightMultiThread(
            outInterUp,
            inp,
            upProj,
            static_cast<const T *>(nullptr),
            B,
            H,
            C,
            interSize);
#pragma omp parallel for collapse(3) num_threads(THREADS_NUM)
        for (size_t b = 0; b < B; b++)
        {
            for (size_t h = 0; h < H; h++)
            {
                for (size_t i = 0; i < interSize; i++)
                {
                    T *outInterGateBH = outInterGate + b * H * interSize + h * interSize;
                    T *outInterUpBH = outInterUp + b * H * interSize + h * interSize;
                    T value = outInterGateBH[i];
                    value *= 1 / (1 + std::exp(-value));
                    outInterGateBH[i] = value * outInterUpBH[i];
                }
            }
        }
        matmulWeightMultiThread(
            out,
            outInterGate,
            downProj,
            static_cast<const T *>(nullptr),
            B,
            H,
            interSize,
            C);
    }
}