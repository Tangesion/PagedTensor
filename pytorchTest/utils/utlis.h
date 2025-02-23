#pragma once
#include <torch/extension.h>
#include "runtime/tensor.h"

namespace paged_tensor::utils
{
    inline runtime::Tensor::UniquePtr torchTopaged_tensor(torch::Tensor &tensor)
    {
        void *data = tensor.data_ptr();
        auto shape = tensor.sizes();
        std::vector<int64_t> shapeVec(shape.begin(), shape.end());
        runtime::Tensor::Shape shapepaged_tensor = runtime::Tensor::makeShape(shapeVec);
        runtime::Tensor::UniquePtr paged_tensorTensor = runtime::Tensor::wrap(data, runtime::Tensor::DataType::kFLOAT, shapepaged_tensor);
        return paged_tensorTensor;
    }

    inline torch::Tensor paged_tensorToTorch(runtime::Tensor::UniquePtr &tensor)
    {
        auto shape = tensor->getShape();
        std::vector<int64_t> shapeVec(shape.d, shape.d + shape.nbDims);
        torch::Tensor torchTensor = torch::from_blob(tensor->data(), shapeVec, torch::kFloat);
        return torchTensor;
    }
}