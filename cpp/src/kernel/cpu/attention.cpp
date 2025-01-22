#include "kernel/cpu/attention.h"

namespace inference_frame::kernel::cpu
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
                        float maxValue = std::numeric_limits<float>::min();
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
                            if (sum > maxValue)
                            {
                                maxValue = sum;
                            }
                            interAttnBNH[h2] = sum / std::sqrt(D);
                        }
                        float minValue = std::numeric_limits<float>::min();
                        for (size_t h2 = h + 1; h2 < H; h2++)
                        {
                            interAttnBNH[h2] = minValue;
                        }
                        // softmax
                        float expSum = 0;
                        for (size_t h2 = 0; h2 <= h; h2++)
                        {
                            float expValue = std::exp(interAttnBNH[h2] - maxValue);
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
                    float maxValue = std::numeric_limits<float>::min();
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
                        if (sum > maxValue)
                        {
                            maxValue = sum;
                        }
                        interAttnBN[h] = sum / std::sqrt(D);
                    }
                    // softmax
                    float expSum = 0;
                    for (size_t h = 0; h < H; h++)
                    {
                        float expValue = std::exp(interAttnBN[h] - maxValue);
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
                        float maxValue = std::numeric_limits<float>::min();
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
                            if (sum > maxValue)
                            {
                                maxValue = sum;
                            }
                            interAttnBNH[h2] = sum / std::sqrt(D);
                        }
                        float minValue = std::numeric_limits<float>::min();
                        for (size_t h2 = h + 1; h2 < H; h2++)
                        {
                            interAttnBNH[h2] = minValue;
                        }
                        // softmax
                        float expSum = 0;
                        for (size_t h2 = 0; h2 <= h; h2++)
                        {
                            float expValue = std::exp(interAttnBNH[h2] - maxValue);
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
                    float maxValue = std::numeric_limits<float>::min();
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
                        if (sum > maxValue)
                        {
                            maxValue = sum;
                        }
                        interAttnBN[h] = sum / std::sqrt(D);
                    }
                    // softmax
                    float expSum = 0;
                    for (size_t h = 0; h < H; h++)
                    {
                        float expValue = std::exp(interAttnBN[h] - maxValue);
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

    // using namespace inference_frame::runtime;
    // using namespace inference_frame::func;
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

} // namespace inference_frame::kernel::cpu