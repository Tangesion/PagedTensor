#include "kernel/cpu/attention.h"

namespace paged_tensor::kernel::cpu
{

    // prefill
    // query key value (B, NH, H, D)
    // out (B, NH, H, D)
    // decode
    // query(B, NH, 1, D)
    // key(B, NH, H, D)
    // value(B, NH, H, D)

    void attentionForwardMultiThread(
        float *out, const float *query, const float *key, const float *value, float *interAttn,
        bool isPrefill,
        const size_t B, const size_t NH, const size_t H, const size_t D)
    {
        if (__builtin_expect(isPrefill, 0))
        {
#pragma omp parallel for collapse(3) num_threads(THREADS_NUM)
            for (size_t b = 0; b < B; b++)
            {
                for (size_t nh = 0; nh < NH; nh++)
                {
                    for (size_t h = 0; h < H; h++)
                    {
                        // q @ k
                        float maxValue = -std::numeric_limits<float>::infinity();
                        float *interAttnBNH = interAttn + b * NH * H * H + nh * H * H + h * H;
                        const float *queryBNH = query + b * NH * H * D + nh * H * D + h * D;
                        for (size_t h2 = 0; h2 <= h; h2++)
                        {
                            float sum = 0;
                            const float *keyBNH = key + b * NH * H * D + nh * H * D + h2 * D;
                            for (size_t d = 0; d < D; d++)
                            {
                                sum += queryBNH[d] * keyBNH[d];
                            }
                            sum /= sqrtf(static_cast<float>(D));
                            if (sum > maxValue)
                            {
                                maxValue = sum;
                            }
                            interAttnBNH[h2] = sum;
                        }
                        float minValue = -std::numeric_limits<float>::infinity();
                        for (size_t h2 = h + 1; h2 < H; h2++)
                        {
                            interAttnBNH[h2] = minValue;
                        }
                        // softmax
                        float expSum = 0;
                        for (size_t h2 = 0; h2 <= h; h2++)
                        {
                            float expValue = expf(interAttnBNH[h2] - maxValue);
                            expSum += expValue;
                            interAttnBNH[h2] = expValue;
                        }
                        float invExpSum = expSum == 0 ? 0 : 1 / expSum;
                        for (size_t h2 = 0; h2 < H; h2++)
                        {
                            if (h2 <= h)
                            {
                                interAttnBNH[h2] *= invExpSum;
                            }
                            else
                            {
                                interAttnBNH[h2] = 0;
                            }
                        }
                        // score @ value
                        float *outBNH = out + b * NH * H * D + nh * H * D + h * D;
                        for (size_t d = 0; d < D; d++)
                        {
                            float sum = 0;
                            for (size_t h2 = 0; h2 < H; h2++)
                            {
                                sum += interAttnBNH[h2] * value[b * NH * H * D + nh * H * D + h2 * D + d];
                            }
                            outBNH[d] = sum;
                        }
                    }
                }
            }
        }
        else
        {
#pragma omp parallel for collapse(2) num_threads(THREADS_NUM)
            // decode
            for (size_t b = 0; b < B; b++)
            {
                for (size_t nh = 0; nh < NH; nh++)
                {
                    float maxValue = -std::numeric_limits<float>::infinity();
                    float *interAttnBN = interAttn + b * NH * H + nh * H;
                    for (size_t h = 0; h < H; h++)
                    {
                        // q @ k
                        const float *queryBN = query + b * NH * D + nh * D;
                        const float *keyBNH = key + b * NH * H * D + nh * H * D + h * D;
                        float sum = 0;
                        for (size_t d = 0; d < D; d++)
                        {
                            sum += queryBN[d] * keyBNH[d];
                        }
                        sum /= sqrtf(static_cast<float>(D));
                        if (sum > maxValue)
                        {
                            maxValue = sum;
                        }
                        interAttnBN[h] = sum;
                    }
                    // softmax
                    float expSum = 0;
                    for (size_t h = 0; h < H; h++)
                    {
                        float expValue = expf(interAttnBN[h] - maxValue);
                        expSum += expValue;
                        interAttnBN[h] = expValue;
                    }
                    float invExpSum = expSum == 0 ? 0 : 1 / expSum;
                    for (size_t h = 0; h < H; h++)
                    {
                        interAttnBN[h] *= invExpSum;
                    }
                    // score @ value
                    float *outBN = out + b * NH * D + nh * D;
                    const float *valueBN = value + b * NH * H * D + nh * H * D;
                    for (size_t d = 0; d < D; d++)
                    {
                        float sum = 0;
                        for (size_t h = 0; h < H; h++)
                        {
                            sum += interAttnBN[h] * valueBN[h * D + d];
                        }
                        outBN[d] = sum;
                    }
                }
            }
        }
    }

    void attentionForwardOneThread(
        float *out, const float *query, const float *key, const float *value, float *interAttn,
        bool isPrefill,
        const size_t B, const size_t NH, const size_t H, const size_t D)
    {
        if (__builtin_expect(isPrefill, 0))
        {
            for (size_t b = 0; b < B; b++)
            {
                for (size_t nh = 0; nh < NH; nh++)
                {
                    for (size_t h = 0; h < H; h++)
                    {
                        // q @ k
                        float maxValue = -std::numeric_limits<float>::infinity();
                        float *interAttnBNH = interAttn + b * NH * H * H + nh * H * H + h * H;
                        const float *queryBNH = query + b * NH * H * D + nh * H * D + h * D;
                        for (size_t h2 = 0; h2 <= h; h2++)
                        {
                            float sum = 0;
                            const float *keyBNH = key + b * NH * H * D + nh * H * D + h2 * D;
                            for (size_t d = 0; d < D; d++)
                            {
                                sum += queryBNH[d] * keyBNH[d];
                            }
                            sum /= sqrtf(static_cast<float>(D));
                            if (sum > maxValue)
                            {
                                maxValue = sum;
                            }
                            interAttnBNH[h2] = sum;
                        }
                        float minValue = -std::numeric_limits<float>::infinity();
                        for (size_t h2 = h + 1; h2 < H; h2++)
                        {
                            interAttnBNH[h2] = minValue;
                        }
                        // softmax
                        float expSum = 0;
                        for (size_t h2 = 0; h2 <= h; h2++)
                        {
                            float expValue = expf(interAttnBNH[h2] - maxValue);
                            expSum += expValue;
                            interAttnBNH[h2] = expValue;
                        }
                        float invExpSum = expSum == 0 ? 0 : 1 / expSum;
                        for (size_t h2 = 0; h2 < H; h2++)
                        {
                            if (h2 <= h)
                            {
                                interAttnBNH[h2] *= invExpSum;
                            }
                            else
                            {
                                interAttnBNH[h2] = 0;
                            }
                        }
                        // score @ value
                        float *outBNH = out + b * NH * H * D + nh * H * D + h * D;
                        for (size_t d = 0; d < D; d++)
                        {
                            float sum = 0;
                            for (size_t h2 = 0; h2 < H; h2++)
                            {
                                sum += interAttnBNH[h2] * value[b * NH * H * D + nh * H * D + h2 * D + d];
                            }
                            outBNH[d] = sum;
                        }
                    }
                }
            }
        }
        else
        {
            // decode
            for (size_t b = 0; b < B; b++)
            {
                for (size_t nh = 0; nh < NH; nh++)
                {
                    float maxValue = -std::numeric_limits<float>::infinity();
                    float *interAttnBN = interAttn + b * NH * H + nh * H;
                    for (size_t h = 0; h < H; h++)
                    {
                        // q @ k
                        const float *queryBN = query + b * NH * D + nh * D;
                        const float *keyBNH = key + b * NH * H * D + nh * H * D + h * D;
                        float sum = 0;
                        for (size_t d = 0; d < D; d++)
                        {
                            sum += queryBN[d] * keyBNH[d];
                        }
                        sum /= sqrtf(static_cast<float>(D));
                        if (sum > maxValue)
                        {
                            maxValue = sum;
                        }
                        interAttnBN[h] = sum;
                    }
                    // softmax
                    float expSum = 0;
                    for (size_t h = 0; h < H; h++)
                    {
                        float expValue = expf(interAttnBN[h] - maxValue);
                        expSum += expValue;
                        interAttnBN[h] = expValue;
                    }
                    float invExpSum = expSum == 0 ? 0 : 1 / expSum;
                    for (size_t h = 0; h < H; h++)
                    {
                        interAttnBN[h] *= invExpSum;
                    }
                    // score @ value
                    float *outBN = out + b * NH * D + nh * D;
                    const float *valueBN = value + b * NH * H * D + nh * H * D;
                    for (size_t d = 0; d < D; d++)
                    {
                        float sum = 0;
                        for (size_t h = 0; h < H; h++)
                        {
                            sum += interAttnBN[h] * valueBN[h * D + d];
                        }
                        outBN[d] = sum;
                    }
                }
            }
        }
    }
    // q (B, H, NH, D)
    void attentionForwardPaged(
        float *out, const float *query, const DataPtr key, const DataPtr value, float *internAttn,
        bool isPrefill,
        const size_t B, const size_t NH, const size_t H, const size_t D)
    {
        if (__builtin_expect(isPrefill, 0))
        {
            for (size_t b = 0; b < B; b++)
            {
                for (size_t nh = 0; nh < NH; nh++)
                {
                    for (size_t h = 0; h < H; h++)
                    {
                        float maxValue = -std::numeric_limits<float>::infinity();
                        float *internAttnBNH = internAttn + b * NH * H * H + nh * H * H + h * H;
                        const float *queryBNH = query + b * NH * H * D + h * NH * D + nh * D;
                        for (size_t h2 = 0; h2 <= h; h2++)
                        {
                            float sum = 0.0;
                            // a block size is (4096(dimmension))
                            DataPtr keyBH = key + b * H * NH * D + h2 * NH * D;
                            const float *keyBNH = keyBH.data<float>() + nh * D;
                            for (size_t d = 0; d < D; d++)
                            {
                                sum += queryBNH[d] * keyBNH[d];
                            }
                            sum /= sqrtf(static_cast<float>(D));
                            if (sum > maxValue)
                            {
                                maxValue = sum;
                            }
                            internAttnBNH[h2] = sum;
                        }
                        float minValue = -std::numeric_limits<float>::infinity();
                        for (size_t h2 = h + 1; h2 < H; h2++)
                        {
                            internAttnBNH[h2] = minValue;
                        }
                        // softmax
                        float expSum = 0;
                        for (size_t h2 = 0; h2 <= h; h2++)
                        {
                            float expValue = expf(internAttnBNH[h2] - maxValue);
                            expSum += expValue;
                            internAttnBNH[h2] = expValue;
                            // std::cout << internAttnBNH[h2] << " ";
                        }
                        // std::cout << std::endl;
                        float invExpSum = expSum == 0 ? 0 : 1 / expSum;
                        for (size_t h2 = 0; h2 < H; h2++)
                        {
                            if (h2 <= h)
                            {
                                internAttnBNH[h2] *= invExpSum;
                            }
                            else
                            {
                                internAttnBNH[h2] = 0;
                            }
                        }
                        // score @ value
                        float *outBNH = out + b * H * NH * D + h * NH * D + nh * D;

                        for (size_t h2 = 0; h2 < H; h2++)
                        {
                            DataPtr valueBH = value + b * H * NH * D + h2 * NH * D;
                            float *valueBNH = valueBH.data<float>() + nh * D;

                            for (size_t d = 0; d < D; d++)
                            {

                                outBNH[d] += internAttnBNH[h2] * valueBNH[d];
                                // std::cout << outBNH[d] << " ";
                            }
                            // std::cout << internAttnBNH[h2] << " ";
                        }
                        // std::cout << std::endl;
                    }
                }
            }
        }
        else
        {
            // decode
            for (size_t b = 0; b < B; b++)
            {
                for (size_t nh = 0; nh < NH; nh++)
                {
                    float maxValue = -std::numeric_limits<float>::infinity();
                    float *internAttnBN = internAttn + b * NH * H + nh * H;
                    const float *queryBN = query + b * NH * D + nh * D;
                    for (size_t h = 0; h < H; h++)
                    {
                        float sum = 0.0;
                        DataPtr keyBH = key + b * H * NH * D + h * NH * D;
                        const float *keytBNH = keyBH.data<float>() + nh * D;
                        for (size_t d = 0; d < D; d++)
                        {
                            sum += queryBN[d] * keytBNH[d];
                        }
                        sum /= sqrtf(static_cast<float>(D));
                        if (sum > maxValue)
                        {
                            maxValue = sum;
                        }
                        internAttnBN[h] = sum;
                    }
                    // softmax
                    float expSum = 0;
                    for (size_t h = 0; h < H; h++)
                    {
                        float expValue = expf(internAttnBN[h] - maxValue);
                        expSum += expValue;
                        internAttnBN[h] = expValue;
                    }
                    float invExpSum = expSum == 0 ? 0 : 1 / expSum;
                    for (size_t h = 0; h < H; h++)
                    {
                        internAttnBN[h] *= invExpSum;
                    }
                    // score @ value
                    float *outBN = out + b * NH * D + nh * D;
                    for (size_t h = 0; h < H; h++)
                    {
                        DataPtr valueBH = value + b * H * NH * D + h * NH * D;
                        const float *valueBNH = valueBH.data<float>() + nh * D;
                        for (size_t d = 0; d < D; d++)
                        {
                            outBN[d] += internAttnBN[h] * valueBNH[d];
                        }
                    }
                }
            }
        }
    }

    /* void attentionForwardPaged(
        DataPtr out, const DataPtr query, const DataPtr key, const DataPtr value, DataPtr interAttn,
        bool isPrefill,
        const size_t B, const size_t NH, const size_t H, const size_t D)
    {
        if (__builtin_expect(isPrefill, 0))
        {
            for (size_t b = 0; b < B; b++)
            {
                for (size_t nh = 0; nh < NH; nh++)
                {
                    for (size_t h = 0; h < H; h++)
                    {
                        // q @ k
                        float maxValue = -std::numeric_limits<float>::infinity();
                        DataPtr interAttnBNH = interAttn + b * NH * H * H + nh * H * H + h * H;
                        const DataPtr queryBNH = query + b * NH * H * D + nh * H * D + h * D;
                        for (size_t h2 = 0; h2 <= h; h2++)
                        {
                            float sum = 0;
                            const DataPtr keyBNH = key + b * NH * H * D + nh * H * D + h2 * D;
                            for (size_t d = 0; d < D; d++)
                            {
                                sum += *(queryBNH[d].data<float>()) * *(keyBNH[d].data<float>());
                            }
                            sum /= sqrtf(static_cast<float>(D));
                            if (sum > maxValue)
                            {
                                maxValue = sum;
                            }
                            *(interAttnBNH[h2].data<float>()) = sum;
                        }
                        float minValue = -std::numeric_limits<float>::infinity();
                        for (size_t h2 = h + 1; h2 < H; h2++)
                        {
                            *(interAttnBNH[h2].data<float>()) = minValue;
                        }
                        // softmax
                        float expSum = 0;
                        for (size_t h2 = 0; h2 <= h; h2++)
                        {
                            float expValue = expf(*(interAttnBNH[h2].data<float>()) - maxValue);
                            expSum += expValue;
                            *(interAttnBNH[h2].data<float>()) = expValue;
                        }
                        float invExpSum = expSum == 0 ? 0 : 1 / expSum;
                        for (size_t h2 = 0; h2 < H; h2++)
                        {
                            if (h2 <= h)
                            {
                                *(interAttnBNH[h2].data<float>()) *= invExpSum;
                            }
                            else
                            {
                                *(interAttnBNH[h2].data<float>()) = 0;
                            }
                        }
                        // score @ value
                        DataPtr outBNH = out + b * NH * H * D + nh * H * D + h * D;
                        for (size_t d = 0; d < D; d++)
                        {
                            float sum = 0;
                            for (size_t h2 = 0; h2 < H; h2++)
                            {
                                sum += *(interAttnBNH[h2].data<float>()) * *(value[b * NH * H * D + nh * H * D + h2 * D + d].data<float>());
                            }
                            *(outBNH[d].data<float>()) = sum;
                        }
                    }
                }
            }
        }
        else
        {
            // decode
            for (size_t b = 0; b < B; b++)
            {
                for (size_t nh = 0; nh < NH; nh++)
                {
                    float maxValue = -std::numeric_limits<float>::infinity();
                    DataPtr interAttnBN = interAttn + b * NH * H + nh * H;
                    for (size_t h = 0; h < H; h++)
                    {
                        // q @ k
                        const DataPtr queryBN = query + b * NH * D + nh * D;
                        const DataPtr keyBNH = key + b * NH * H * D + nh * H * D + h * D;
                        float sum = 0;
                        for (size_t d = 0; d < D; d++)
                        {
                            sum += *(queryBN[d].data<float>()) * *(keyBNH[d].data<float>());
                        }
                        sum /= sqrtf(static_cast<float>(D));
                        if (sum > maxValue)
                        {
                            maxValue = sum;
                        }
                        *(interAttnBN[h].data<float>()) = sum;
                    }
                    // softmax
                    float expSum = 0;
                    for (size_t h = 0; h < H; h++)
                    {
                        float expValue = expf(*(interAttnBN[h].data<float>()) - maxValue);
                        expSum += expValue;
                        *(interAttnBN[h].data<float>()) = expValue;
                    }
                    float invExpSum = expSum == 0 ? 0 : 1 / expSum;
                    for (size_t h = 0; h < H; h++)
                    {
                        *(interAttnBN[h].data<float>()) *= invExpSum;
                    }
                    // score @ value
                    DataPtr outBN = out + b * NH * D + nh * D;
                    const DataPtr valueBN = value + b * NH * H * D + nh * H * D;
                    for (size_t d = 0; d < D; d++)
                    {
                        float sum = 0;
                        for (size_t h = 0; h < H; h++)
                        {
                            sum += *(interAttnBN[h].data<float>()) * *(valueBN[h * D + d].data<float>());
                        }
                        *(outBN[d].data<float>()) = sum;
                    }
                }
            }
        }
    } */

    // using namespace paged_tensor::runtime;
    // using namespace paged_tensor::func;
    //
    // void attentionForwardLaunch(
    //    Tensor::SharedPtr out, const Tensor::SharedPtr query, const Tensor::SharedPtr key, const Tensor::SharedPtr value, Tensor::SharedPtr interAttn,
    //    bool isPrefill)
    //{
    //    DataType dataType = out->getDataType();
    //    size_t B = out->getShape().d[0];
    //    size_t NH = out->getShape().d[1];
    //    size_t H = out->getShape().d[2];
    //    size_t D = out->getShape().d[3];
    //    if (dataType == DataType::kFLOAT)
    //    {
    //        attentionForward<float>(
    //            getData<Tensor::DataType::kFLOAT>(out), getData<Tensor::DataType::kFLOAT>(query), getData<Tensor::DataType::kFLOAT>(key), getData<Tensor::DataType::kFLOAT>(value), getData<Tensor::DataType::kFLOAT>(interAttn),
    //            isPrefill,
    //            B, NH, H, D);
    //    }
    //}

} // namespace paged_tensor::kernel::cpu