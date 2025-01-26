#pragma once
#include <torch/extension.h>
#include "runtime/tensor.h"

namespace toy::utils
{
    inline runtime::Tensor::UniquePtr torchToToy(torch::Tensor &tensor)
    {
        void *data = tensor.data_ptr();
        auto shape = tensor.sizes();
        std::vector<int64_t> shapeVec(shape.begin(), shape.end());
        runtime::Tensor::Shape shapeToy = runtime::Tensor::makeShape(shapeVec);
        runtime::Tensor::UniquePtr toyTensor = runtime::Tensor::wrap(data, runtime::Tensor::DataType::kFLOAT, shapeToy);
        return toyTensor;
    }

    inline torch::Tensor toyToTorch(runtime::Tensor::UniquePtr &tensor)
    {
        auto shape = tensor->getShape();
        std::vector<int64_t> shapeVec(shape.d, shape.d + shape.nbDims);
        torch::Tensor torchTensor = torch::from_blob(tensor->data(), shapeVec, torch::kFloat);
        return torchTensor;
    }
}