#pragma once
#include "kernel/cpu/rope.h"
#include "runtime/tensor.h"

namespace inference_frame::kernel::launch
{
    namespace kernel_cpu = inference_frame::kernel::cpu;
    using UniquePtr = inference_frame::runtime::Tensor::UniquePtr;
    using UniquePtrConst = inference_frame::runtime::Tensor::UniqueConstPtr;
    using DataType = inference_frame::runtime::Tensor::DataType;

    void precomputeFreqsCosSin(UniquePtr &freqsCosSin, const size_t dim, const size_t maxPos, const float theta = 10000.0, const bool isMultiThread = true)
    {
        DataType dataTypeFreqsCosSin = freqsCosSin->getDataType();
        try
        {
            CHECK_WITH_INFO(dataTypeFreqsCosSin == DataType::kFLOAT, "Data type of freqsCosSin must be float");
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            std::exit(EXIT_FAILURE);
        }

        int64_t D = freqsCosSin->getShape().d[2];

        switch (dataTypeFreqsCosSin)
        {
        case DataType::kFLOAT:
        {
            auto *freqsCosSinData = static_cast<float *>(freqsCosSin->data());
            if (isMultiThread)
            {
                kernel_cpu::precomputeFreqsCosSinMultiThread(freqsCosSinData, dim, maxPos, theta);
            }
            else
            {
                kernel_cpu::precomputeFreqsCosSinOneThread(freqsCosSinData, dim, maxPos, theta);
            }
            break;
        }
        default:
            break;
        }
    }
    void applyRope(UniquePtr &out, UniquePtr &inp, UniquePtr &freqsCosSin, UniquePtr &pos, const bool isMultiThread = true)
    {
        DataType dataTypeOut = out->getDataType();
        DataType dataTypeInp = inp->getDataType();
        DataType dataTypeFreqsCosSin = freqsCosSin->getDataType();
        DataType dataTypePos = pos->getDataType();
        try
        {
            CHECK_WITH_INFO(dataTypeOut == dataTypeInp, "Data type of output and input must be the same");
            CHECK_WITH_INFO(dataTypeOut == dataTypeFreqsCosSin, "Data type of output and freqsCosSin must be the same");
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            std::exit(EXIT_FAILURE);
        }

        int64_t B = out->getShape().d[0];
        int64_t NH = out->getShape().d[1];
        int64_t H = out->getShape().d[2];
        int64_t D = out->getShape().d[3];

        switch (dataTypeOut)
        {
        case DataType::kFLOAT:
        {
            auto *outData = static_cast<float *>(out->data());
            auto *inpData = static_cast<float *>(inp->data());
            auto *freqsCosSinData = static_cast<float *>(freqsCosSin->data());
            auto *posData = static_cast<size_t *>(pos->data());
            if (isMultiThread)
            {
                kernel_cpu::applyRopeMultiThread(outData, inpData, freqsCosSinData, B, NH, H, D, posData);
            }
            else
            {
                kernel_cpu::applyRopeOneThread(outData, inpData, freqsCosSinData, B, NH, H, D, posData);
            }
            break;
        }
        default:
            break;
        }
    }
}