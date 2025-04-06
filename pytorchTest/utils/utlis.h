#pragma once
#include <torch/extension.h>
#include "func/func.h"

namespace paged_tensor::utils
{
    inline runtime::Tensor::UniquePtr torchToPagedTensor(torch::Tensor &tensor, bool paged = false)
    {
        if (!paged)
        {
            void *data = tensor.data_ptr();
            auto shape = tensor.sizes();
            std::vector<int64_t> shapeVec(shape.begin(), shape.end());
            runtime::Tensor::Shape shapePagedTensor = runtime::Tensor::makeShape(shapeVec);
            runtime::Tensor::UniquePtr pagedTensor = runtime::Tensor::wrap(data, runtime::Tensor::DataType::kFLOAT, shapePagedTensor);
            return pagedTensor;
        }
        else
        {
            void *data = tensor.data_ptr();
            auto shape = tensor.sizes();
            std::vector<int64_t> shapeVec(shape.begin(), shape.end());
            runtime::Tensor::Shape shapePagedTensor = runtime::Tensor::makeShape(shapeVec);
            static runtime::Tensor::UniquePtr continuousTensor = runtime::Tensor::wrap(data, runtime::Tensor::DataType::kFLOAT, shapePagedTensor);
            return func::continuousToPaged(continuousTensor);
        }
    }

    inline torch::Tensor pagedTensorToTorch(runtime::Tensor::UniquePtr &tensor)
    {
        if (!tensor->isPaged())
        {
            auto shape = tensor->getShape();
            std::vector<int64_t> shapeVec(shape.d, shape.d + shape.nbDims);
            torch::Tensor torchTensor = torch::from_blob(tensor->data(), shapeVec, torch::kFloat);
            return torchTensor;
        }
        else
        {
            static runtime::Tensor::UniquePtr continuousTensor = func::pagedToContinuous(tensor);
            auto shape = continuousTensor->getShape();
            std::vector<int64_t> shapeVec(shape.d, shape.d + shape.nbDims);
            torch::Tensor torchTensor = torch::from_blob(continuousTensor->data(), shapeVec, torch::kFloat);
            return torchTensor;
        }
    }
}