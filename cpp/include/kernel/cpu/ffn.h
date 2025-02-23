#pragma once
#include <cmath>
#include <iostream>
#include "matmul.h"

// down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
namespace paged_tensor::kernel::cpu
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
        const size_t interSize);

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
        const size_t interSize);
}