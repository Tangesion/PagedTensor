#pragma once
#include "kernel/cpu/transpose.h"
#include "func/func.h"

namespace inference_frame::kernel::launch
{
    namespace kernel_cpu = inference_frame::kernel::cpu;
    using UniquePtr = inference_frame::runtime::Tensor::UniquePtr;
    using DataType = inference_frame::runtime::Tensor::DataType;
    using MemoryType = inference_frame::runtime::MemoryType;

    void transposeOneThread(UniquePtr &out, UniquePtr &inp)
    {
        DataType dataTypeOut = out->getDataType();
        DataType dataTypeInp = inp->getDataType();
        try
        {
            CHECK_WITH_INFO(dataTypeOut == dataTypeInp, "Data type of output and input must be the same");
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            std::exit(EXIT_FAILURE);
        }
        int64_t B = inp->getShape().d[0];
        int64_t H = inp->getShape().d[1];
        int64_t NH = inp->getShape().d[2];
        int64_t D = inp->getShape().d[3];

        switch (dataTypeOut)
        {
        case DataType::kFLOAT:
        {
            auto *outData = static_cast<float *>(out->data());
            auto *inpData = static_cast<float *>(inp->data());
            kernel_cpu::transpose(inpData, outData, B, NH, H, D);
            break;
        }

        default:
            break;
        }
        inp.reset();
    }

    void transposeMultiThread(UniquePtr &out, UniquePtr &inp)
    {
        DataType dataTypeOut = out->getDataType();
        DataType dataTypeInp = inp->getDataType();
        try
        {
            CHECK_WITH_INFO(dataTypeOut == dataTypeInp, "Data type of output and input must be the same");
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            std::exit(EXIT_FAILURE);
        }
        int64_t B = inp->getShape().d[0];
        int64_t H = inp->getShape().d[1];
        int64_t NH = inp->getShape().d[2];
        int64_t D = inp->getShape().d[3];

        switch (dataTypeOut)
        {
        case DataType::kFLOAT:
        {
            auto *outData = static_cast<float *>(out->data());
            auto *inpData = static_cast<float *>(inp->data());
            kernel_cpu::transposeMultiThread(inpData, outData, B, NH, H, D);
            break;
        }

        default:
            break;
        }
        inp.reset();
    }

    void transpose(UniquePtr &out, UniquePtr &inp, const bool isMultiThread = true)
    {
        if (isMultiThread)
        {
            transposeMultiThread(out, inp);
        }
        else
        {
            transposeOneThread(out, inp);
        }
    }
}