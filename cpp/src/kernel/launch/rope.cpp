#include "kernel/cpu/rope.h"
#include "runtime/tensor.h"

namespace paged_tensor::kernel::launch
{
    namespace kernel_cpu = paged_tensor::kernel::cpu;
    using UniquePtr = paged_tensor::runtime::Tensor::UniquePtr;
    using UniquePtrConst = paged_tensor::runtime::Tensor::UniqueConstPtr;
    using DataType = paged_tensor::runtime::Tensor::DataType;

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
    void applyRope(UniquePtr &inp, UniquePtr &freqsCosSin, UniquePtr &pos, const bool isMultiThread = true)
    {
        DataType dataTypeInp = inp->getDataType();
        DataType dataTypeFreqsCosSin = freqsCosSin->getDataType();
        DataType dataTypePos = pos->getDataType();
        try
        {
            CHECK_WITH_INFO(dataTypeInp == dataTypeFreqsCosSin, "Data type of output and freqsCosSin must be the same");
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            std::exit(EXIT_FAILURE);
        }

        int64_t B = inp->getShape().d[0];
        int64_t NH = inp->getShape().d[2];
        int64_t H = inp->getShape().d[1];
        int64_t D = inp->getShape().d[3];

        if (!inp->isPaged())
        {
            switch (dataTypeInp)
            {
            case DataType::kFLOAT:
            {
                auto *inpData = static_cast<float *>(inp->data());
                auto *freqsCosSinData = static_cast<float *>(freqsCosSin->data());
                auto *posData = static_cast<size_t *>(pos->data());
                if (isMultiThread)
                {
                    kernel_cpu::applyRopeMultiThread(inpData, freqsCosSinData, B, NH, H, D, posData);
                }
                else
                {
                    kernel_cpu::applyRopeOneThread(inpData, freqsCosSinData, B, NH, H, D, posData);
                }
                break;
            }
            default:
                break;
            }
        }
        else
        {
            switch (dataTypeInp)
            {
            case DataType::kFLOAT:
            {
                DataPtr inpData = inp->dataPaged();
                auto *freqsCosSinData = static_cast<float *>(freqsCosSin->data());
                auto *posData = static_cast<size_t *>(pos->data());
                kernel_cpu::applyRopePagedOneThread(inpData, freqsCosSinData, B, NH, H, D, posData);
                break;
            }
            default:
                break;
            }
        }
    }
}