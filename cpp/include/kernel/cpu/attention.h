#pragma once
#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <limits>
#include <memory>
// #include "func/threadPool.h"
// #include "func/func.h"
// #include "runtime/tensor.h"

#define THREADS_NUM 28

namespace inference_frame::kernel::cpu
{
    // prefill
    // query key value (B, NH, H, D)
    // out (B, NH, H, D)
    // decode
    // query(B, NH, 1, D)
    // key(B, NH, H, D)
    // value(B, NH, H, D)

    template <typename T>
    void attentionForward(
        T *out, const T *query, const T *key, const T *value, T *interAttn,
        bool isPrefill,
        const size_t B, const size_t NH, const size_t H, const size_t D)
    {
        if (isPrefill)
        {
            for (size_t b = 0; b < B; b++)
            {
                for (size_t nh = 0; nh < NH; nh++)
                {
                    for (size_t h = 0; h < H; h++)
                    {
                        // q @ k
                        T maxValue = std::numeric_limits<T>::min();
                        T *interAttnBNH = interAttn + b * NH * H * H + nh * H * H + h * H;
                        const T *queryBNH = query + b * NH * H * D + nh * H * D + h * D;
                        for (size_t h2 = 0; h2 <= h; h2++)
                        {
                            T sum = 0;
                            const T *keyBNH = key + b * NH * H * D + nh * H * D + h2 * D;
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
                        T minValue = std::numeric_limits<T>::min();
                        for (size_t h2 = h + 1; h2 < H; h2++)
                        {
                            interAttnBNH[h2] = minValue;
                        }
                        // softmax
                        T expSum = 0;
                        for (size_t h2 = 0; h2 <= h; h2++)
                        {
                            T expValue = std::exp(interAttnBNH[h2] - maxValue);
                            expSum += expValue;
                            interAttnBNH[h2] = expValue;
                        }
                        T invExpSum = expSum == 0 ? 0 : 1 / expSum;
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
                        T *outBNH = out + b * NH * H * D + nh * H * D + h * D;
                        for (size_t d = 0; d < D; d++)
                        {
                            T sum = 0;
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