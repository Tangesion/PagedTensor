#pragma once
#include "kernel/cpu/attention.h"
#include "runtime/tensor.h"
#include "func/func.h"

namespace inference_frame::kernel::launch
{
    namespace kernel_cpu = inference_frame::kernel::cpu;
    using SharedPtr = inference_frame::runtime::Tensor::SharedPtr;
    using DataType = inference_frame::runtime::Tensor::DataType;

    void attentionForward(SharedPtr out, SharedPtr query, SharedPtr key, SharedPtr value, SharedPtr interAttn, bool isPrefill, cpu::AttentionType attentionType)
    {
        DataType dataTypeOut = out->getDataType();
        DataType dataTypeQuery = query->getDataType();
        DataType dataTypeKey = key->getDataType();
        DataType dataTypeValue = value->getDataType();
        DataType dataTypeInterAttn = interAttn->getDataType();
        try
        {
            CHECK_WITH_INFO(dataTypeOut == dataTypeQuery, "Data type of output and query must be the same");
            CHECK_WITH_INFO(dataTypeOut == dataTypeKey, "Data type of output and key must be the same");
            CHECK_WITH_INFO(dataTypeOut == dataTypeValue, "Data type of output and value must be the same");
            CHECK_WITH_INFO(dataTypeOut == dataTypeInterAttn, "Data type of output and interAttn must be the same");
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

        auto *outData = inference_frame::func::getData<DataType::kFLOAT>(out);
        auto *queryData = inference_frame::func::getData<DataType::kFLOAT>(query);
        auto *keyData = inference_frame::func::getData<DataType::kFLOAT>(key);
        auto *valueData = inference_frame::func::getData<DataType::kFLOAT>(value);
        auto *interAttnData = inference_frame::func::getData<DataType::kFLOAT>(interAttn);

        switch (attentionType)
        {
        case kernel_cpu::AttentionType::kAttentionOneThread:
            kernel_cpu::attentionForwardOneThread(outData, queryData, keyData, valueData, interAttnData, isPrefill, B, NH, H, D);
            break;
        case kernel_cpu::AttentionType::kAttentionMultiThread:
            kernel_cpu::attentionForwardMultiThread(outData, queryData, keyData, valueData, interAttnData, isPrefill, B, NH, H, D);
            break;
        }
    }

} // namespace inference_frame::kernel::launch