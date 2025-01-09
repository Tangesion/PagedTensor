#pragma once
#include "kernel/cpu/rmsnorm.h"
#include "runtime/tensor.h"

namespace inference_frame::kernel::launch
{
    namespace kernel_cpu = inference_frame::kernel::cpu;
    using UniquePtr = inference_frame::runtime::Tensor::UniquePtr;
    using DataType = inference_frame::runtime::Tensor::DataType;

    void rmsNorm(UniquePtr &out, UniquePtr &inp, UniquePtr &weight, const bool isMultiThread)
    {
        DataType dataTypeOut = out->getDataType();
        DataType dataTypeInp = inp->getDataType();
        DataType dataTypeWeight = weight->getDataType();
        try
        {
            CHECK_WITH_INFO(dataTypeOut == dataTypeInp, "Data type of output and input must be the same");
            CHECK_WITH_INFO(dataTypeOut == dataTypeWeight, "Data type of output and weight must be the same");
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            std::exit(EXIT_FAILURE);
        }

        int64_t B = out->getShape().d[0];
        int64_t H = out->getShape().d[1];
        int64_t C = out->getShape().d[2];

        switch (dataTypeOut)
        {
        case DataType::kFLOAT:
        {
            auto *outData = static_cast<float *>(out->data());
            auto *inpData = static_cast<float *>(inp->data());
            auto *weightData = static_cast<float *>(weight->data());
            if (isMultiThread)
            {
                kernel_cpu::rmsNormMultiThread(outData, inpData, weightData, B, H, C);
            }
            else
            {
                kernel_cpu::rmsNormOneThread(outData, inpData, weightData, B, H, C);
            }
            break;
        }
        default:
            break;
        }
    }
}
