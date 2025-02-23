#pragma once
#include "runtime/tensor.h"
#include "runtime/buffer.h"
#include "runtime/bufferManager.h"
#include "common/assert.h"
#include <cstdlib>
#include <variant>
// #include <torch/extension.h>

namespace paged_tensor::func
{

    runtime::Tensor::UniquePtr createTensor(std::initializer_list<runtime::Tensor::DimType64> const &dims_list, runtime::Tensor::DataType const &type, runtime::MemoryType device, bool paged = false);

    runtime::Tensor::UniquePtr randTensor(std::initializer_list<runtime::Tensor::DimType64> const &dims_list, runtime::Tensor::DataType const &type, runtime::MemoryType device, bool paged = false);

    void reShape(runtime::Tensor::UniquePtr &tensor, std::initializer_list<runtime::Tensor::DimType64> const &dims_list);

    runtime::Tensor::UniquePtr makeRange(const int64_t start, const int64_t end, const int64_t span, runtime::MemoryType device);

    // runtime::Tensor::UniquePtr torchTopaged_tensor(torch::Tensor &tensor);

    // torch::Tensor paged_tensorToTorch(runtime::Tensor::Tensor::UniquePtr &tensor);

    // template <runtime::Tensor::DataType T>
    // struct DataTypeInfo;

    // template <>
    // struct DataTypeInfo<runtime::Tensor::DataType::kFLOAT>
    // {
    //     using Type = float;
    // };

    // template <>
    // struct DataTypeInfo<runtime::Tensor::DataType::kINT32>
    // {
    //     using Type = int32_t;
    // };

    // using TensorDataVariant = std::variant<float *, int32_t *>;

    // template <runtime::Tensor::DataType T>
    // typename DataTypeInfo<T>::Type *getData(runtime::Tensor::UniquePtr tensor)
    // {
    //     return static_cast<typename DataTypeInfo<T>::Type *>(tensor->data());
    // }

    // TensorDataVariant getData(runtime::Tensor::UniquePtr tensor)
    // {
    //     switch (tensor->getDataType())
    //     {
    //     case runtime::Tensor::DataType::kFLOAT:
    //         return getData<runtime::Tensor::DataType::kFLOAT>(tensor);
    //     case runtime::Tensor::DataType::kINT32:
    //         return getData<runtime::Tensor::DataType::kINT32>(tensor);
    //     default:
    //         JUST_THROW("Unsupported data type");
    //         break;
    //     }
    // }

}