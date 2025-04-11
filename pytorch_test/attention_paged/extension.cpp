#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cstddef>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include "../utils/utlis.h"
#include "kernel/launch/attention.h"

using namespace paged_tensor;

torch::Tensor attentionPrefillBind(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    auto B = q.size(0);
    auto NH = q.size(1);
    auto H = q.size(2);
    auto D = q.size(3);

    torch::Tensor out = torch::zeros_like(q);
    torch::Tensor interAttn = torch::zeros({B, NH, H, H});

    runtime::Tensor::UniquePtr pagedQ = utils::torchToPagedTensor(q, true);
    runtime::Tensor::UniquePtr pagedK = utils::torchToPagedTensor(k, true);
    runtime::Tensor::UniquePtr pagedV = utils::torchToPagedTensor(v, true);
    runtime::Tensor::UniquePtr pagedInterAttn = utils::torchToPagedTensor(interAttn, true);
    runtime::Tensor::UniquePtr pagedO = utils::torchToPagedTensor(out, true);

    kernel::launch::attentionForward(pagedO, pagedQ, pagedK, pagedV, pagedInterAttn, true, kernel::cpu::AttentionType::kAttentionOneThread);
    out = utils::pagedTensorToTorch(pagedO);
    return out;
}

torch::Tensor attentionDecodeBind(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    auto B = q.size(0);
    auto NH = q.size(1);
    auto H = k.size(2);
    auto D = q.size(3);

    torch::Tensor out = torch::zeros_like(q);
    torch::Tensor interAttn = torch::zeros({B, NH, 1, H});

    runtime::Tensor::UniquePtr pagedQ = utils::torchToPagedTensor(q, true);
    runtime::Tensor::UniquePtr pagedK = utils::torchToPagedTensor(k, true);
    runtime::Tensor::UniquePtr pagedV = utils::torchToPagedTensor(v, true);
    runtime::Tensor::UniquePtr pagedInterAttn = utils::torchToPagedTensor(interAttn, true);
    runtime::Tensor::UniquePtr pagedO = utils::torchToPagedTensor(out, true);

    kernel::launch::attentionForward(pagedO, pagedQ, pagedK, pagedV, pagedInterAttn, false, kernel::cpu::AttentionType::kAttentionOneThread);
    out = utils::pagedTensorToTorch(pagedO);
    return out;
}

PYBIND11_MODULE(attention_paged, m)
{
    m.def("forward_prefill", torch::wrap_pybind_function(attentionPrefillBind), "paged attention forward");
    m.def("forward_decode", torch::wrap_pybind_function(attentionDecodeBind), "paged attention forward");
}