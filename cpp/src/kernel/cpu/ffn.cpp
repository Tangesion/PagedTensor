#include "kernel/cpu/ffn.h"

// down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
namespace toy::kernel::cpu
{
    void ffnForwardOneThread(
        float *out,
        const float *inp,
        float *outInterGate,
        float *outInterUp,
        const float *gateProj,
        const float *upProj,
        const float *downProj,
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
            static_cast<const float *>(nullptr),
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
            static_cast<const float *>(nullptr),
            B,
            H,
            C,
            interSize);
        // self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        for (size_t b = 0; b < B; b++)
        {
            for (size_t h = 0; h < H; h++)
            {
                float *outInterGateBH = outInterGate + b * H * interSize + h * interSize;
                float *outInterUpBH = outInterUp + b * H * interSize + h * interSize;
                for (size_t i = 0; i < interSize; i++)
                {
                    float value = outInterGateBH[i];
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
            static_cast<const float *>(nullptr),
            B,
            H,
            interSize,
            C);
    }

    void ffnForwardMultiThread(
        float *out,
        const float *inp,
        float *outInterGate,
        float *outInterUp,
        const float *gateProj,
        const float *upProj,
        const float *downProj,
        const size_t B,
        const size_t H,
        const size_t C,
        const size_t interSize)
    {
        matmulWeightMultiThread(
            outInterGate,
            inp,
            gateProj,
            static_cast<const float *>(nullptr),
            B,
            H,
            C,
            interSize);
        matmulWeightMultiThread(
            outInterUp,
            inp,
            upProj,
            static_cast<const float *>(nullptr),
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
                    float *outInterGateBH = outInterGate + b * H * interSize + h * interSize;
                    float *outInterUpBH = outInterUp + b * H * interSize + h * interSize;
                    float value = outInterGateBH[i];
                    value *= 1 / (1 + std::exp(-value));
                    outInterGateBH[i] = value * outInterUpBH[i];
                }
            }
        }
        matmulWeightMultiThread(
            out,
            outInterGate,
            downProj,
            static_cast<const float *>(nullptr),
            B,
            H,
            interSize,
            C);
    }
}