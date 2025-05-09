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
            runtime::Tensor::UniquePtr continuousTensor = runtime::Tensor::wrap(data, runtime::Tensor::DataType::kFLOAT, shapePagedTensor);
            // auto *rawContinuousTensor = continuousTensor.release();
            return func::continuousToPaged(continuousTensor);
        }
    }

    inline torch::Tensor pagedTensorToTorch(runtime::Tensor::UniquePtr &tensor)
    {
        if (!tensor->isPaged())
        {
            auto shape = tensor->getShape();
            std::vector<int64_t> shapeVec(shape.d, shape.d + shape.nbDims);
            auto *rawTensor = tensor.release();
            torch::Tensor torchTensor = torch::from_blob(rawTensor->data(), shapeVec, torch::kFloat).clone();
            delete rawTensor;
            return torchTensor;
        }
        else
        {
            runtime::Tensor::UniquePtr continuousTensor = func::pagedToContinuous(tensor);
            auto shape = continuousTensor->getShape();
            std::vector<int64_t> shapeVec(shape.d, shape.d + shape.nbDims);
            auto *rawContinuousTensor = continuousTensor.release();
            torch::Tensor torchTensor = torch::from_blob(rawContinuousTensor->data(), shapeVec, torch::kFloat).clone();
            delete rawContinuousTensor;
            return torchTensor;
        }
    }
}