#include <iostream>
#include <cstdlib>
#include <memory>

namespace inference_frame::kernel::cpu
{

    // inp (B, T, C) weight (OC, C) bias (OC) out (B, T, OC)
    template <typename T>
    void matmul(T *out, const T *inp, const T *weight, const T *bias, const size_t B, const size_t T, const size_t C, const size_t OC)
    {

        for (size_t b = 0; b < B; b++)
        {
            for (size_t t = 0; t < T; t++)
            {
                T *outBT = out + b * T * OC + t * OC;
                constexpr T *inpBT = inp + b * T * C + t * C;
                for (size_t oc = 0; oc < OC; oc++)
                {
                    constexpr T *weightRow = weight + oc * C;
                    T sum = 0;
                    for (size_t c = 0; c < C; c++)
                    {
                        sum += inpBT[c] * weightRow[c];
                    }
                    outBT[oc] = sum + bias[oc];
                }
            }
        }
    }

} // namespace inference_frame::kernel::cpu