#include <cuda_runtime.h>

// namespace toy::common
//{
//     static char const *_cudaGetErrorEnum(cudaError_t error)
//     {
//         return cudaGetErrorString(error);
//     }
//
//     template <typename T>
//     void check(T result, char const *const func, char const *const file, int const line)
//     {
//         if (result)
//         {
//             throw TllmException(
//                 file, line, fmtstr("[TensorRT-LLM][ERROR] CUDA runtime error in %s: %s", func, _cudaGetErrorEnum(result)));
//         }
//     }
//
// }
// #def ine  TLLM_CUDA_CHECK(stat)                                              \
//    do                                                                     \
//    {                                                                      \
//        toy::common::check((stat), #stat, __FILE__, __LINE__); \
//    } while (0)
//