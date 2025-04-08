#include "runtime/tensor.h"
#include "kernel/cpu/matmul.h"

namespace paged_tensor::kernel::launch
{
    namespace kernel_cpu = paged_tensor::kernel::cpu;
    using UniquePtr = paged_tensor::runtime::Tensor::UniquePtr;
    using DataType = paged_tensor::runtime::Tensor::DataType;

    void matmulWeight(UniquePtr &out, UniquePtr &inp, UniquePtr &weight, const UniquePtr &bias, const cpu::MatmulType matmulType)
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

        if (out->isPaged())
        {
            switch (dataTypeOut)
            {
            case DataType::kFLOAT:
            {
                DataPtr outData = out->dataPaged();
                DataPtr inpData = inp->dataPaged();
                auto *weightData = static_cast<float *>(weight->data());
                auto *biasData = bias == nullptr ? nullptr : static_cast<float *>(bias->data());
                switch (matmulType)
                {
                case kernel_cpu::MatmulType::kMatmulOneThread:
                    kernel_cpu::matmulWeightPaged(outData, inpData, weightData, biasData, B, H, C, OC);
                    break;
                case kernel_cpu::MatmulType::kMatmulMultiThread:
                    kernel_cpu::matmulWeightPagedMultiThread(outData, inpData, weightData, biasData, B, H, C, OC);
                    break;
                case kernel_cpu::MatmulType::kMatmulBlock:
                    kernel_cpu::matmulWeightPagedBlock(outData, inpData, weightData, biasData, B, H, C, OC);
                }
            }
            }
        }
        else
        {
            switch (dataTypeOut)
            {
            case DataType::kFLOAT:
            {
                auto *outData = static_cast<float *>(out->data());
                auto *inpData = static_cast<float *>(inp->data());
                auto *weightData = static_cast<float *>(weight->data());
                auto *biasData = bias == nullptr ? nullptr : static_cast<float *>(bias->data());
                switch (matmulType)
                {
                case kernel_cpu::MatmulType::kMatmulOneThread:
                    kernel_cpu::matmulWeight(outData, inpData, weightData, biasData, B, H, C, OC);
                    break;
                case kernel_cpu::MatmulType::kMatmulMultiThread:
                    kernel_cpu::matmulWeightMultiThread(outData, inpData, weightData, biasData, B, H, C, OC);
                    break;
                case kernel_cpu::MatmulType::kMatmulThreadPool:
                    kernel_cpu::matmulWeightThreadPool(outData, inpData, weightData, biasData, B, H, C, OC);
                    break;
                }
                break;
            }

            default:
                break;
            }
        }
    }

} // namespace paged_tensor::kernel::launch
