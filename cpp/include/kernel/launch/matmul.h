#pragma once
#include "kernel/cpu/matmul.h"

namespace inference_frame::kernel::launch
{
    namespace kernel_cpu = inference_frame::kernel::cpu;
    using SharedPtr = inference_frame::runtime::Tensor::SharedPtr;
    using DataType = inference_frame::runtime::Tensor::DataType;

    void matmulWeight(SharedPtr out, SharedPtr inp, SharedPtr weight, SharedPtr bias, cpu::MatmulType matmulType)
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

        auto *outData = inference_frame::func::getData<DataType::kFLOAT>(out);
        auto *inpData = inference_frame::func::getData<DataType::kFLOAT>(inp);
        auto *weightData = inference_frame::func::getData<DataType::kFLOAT>(weight);
        auto *biasData = bias == nullptr ? nullptr : inference_frame::func::getData<DataType::kFLOAT>(bias);

        switch (matmulType)
        {
        case kernel_cpu::MatmulType::kMatmulOneThread:
            kernel_cpu::matmulWeight(outData, inpData, weightData, biasData, B, H, C, OC);
            break;
        case kernel_cpu::MatmulType::KMatmulMultiThread:
            kernel_cpu::matmulWeightMultiThread(outData, inpData, weightData, biasData, B, H, C, OC);
            break;
        case kernel_cpu::MatmulType::kMatmulThreadPool:
            kernel_cpu::matmulWeightThreadPool(outData, inpData, weightData, biasData, B, H, C, OC);
            break;
        }
    }

} // namespace inference_frame::kernel::launch
