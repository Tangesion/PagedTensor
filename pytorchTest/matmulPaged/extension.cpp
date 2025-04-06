#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cstddef>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include "../utils/utlis.h"
#include "kernel/launch/matmul.h"

using namespace paged_tensor;

torch::Tensor matmulOneThreadBind(torch::Tensor inp, torch::Tensor weight)
{
    auto B = inp.size(0);
    auto H = inp.size(1);
    auto C = inp.size(2);
    torch::Tensor out = torch::zeros_like(inp);

    runtime::Tensor::UniquePtr pagedInp = utils::torchToPagedTensor(inp, true);
    runtime::Tensor::UniquePtr pagedOut = utils::torchToPagedTensor(out, true);
    runtime::Tensor::UniquePtr continuousWeight = utils::torchToPagedTensor(weight, false);

    kernel::launch::matmulWeight(pagedOut, pagedInp, continuousWeight, nullptr, kernel::cpu::MatmulType::kMatmulOneThread);
    out = utils::pagedTensorToTorch(pagedOut);
    return out;
}

PYBIND11_MODULE(matmulPaged, m)
{
    m.def("matmul", torch::wrap_pybind_function(matmulOneThreadBind), "matmul paged");
}