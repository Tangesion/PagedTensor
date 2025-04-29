#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cstddef>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include "../utils/utlis.h"
#include "kernel/cpu/rope.h"
#include "kernel/launch/rope.h"

using namespace paged_tensor;

torch::Tensor precomputeFreqsCosSinBind(const long dim, const long maxPos, const float theta = 10000.0)
{
    auto freqsCosSin = torch::zeros({maxPos, 2, dim / 2}, torch::kFloat);
    kernel::cpu::precomputeFreqsCosSinMultiThread(freqsCosSin.data_ptr<float>(), dim, maxPos, theta);
    return freqsCosSin;
}

torch::Tensor applyRopePagedBind(torch::Tensor inp, torch::Tensor freqsCosSin, torch::Tensor pos)
{
    auto B = inp.size(0);
    auto NH = inp.size(2);
    auto H = inp.size(1);
    auto D = inp.size(3);

    runtime::Tensor::UniquePtr pagedInp = utils::torchToPagedTensor(inp, true);
    runtime::Tensor::UniquePtr freqsCosSinContinuous = utils::torchToPagedTensor(freqsCosSin, false);
    runtime::Tensor::UniquePtr posContinuous = utils::torchToPagedTensor(pos, false);

    kernel::launch::applyRope(pagedInp, freqsCosSinContinuous, posContinuous);

    inp = utils::pagedTensorToTorch(pagedInp);

    // torch::Tensor out = torch::zeros_like(inp);
    // kernel::cpu::applyRopePagedOneThread(
    //    inp.data_ptr<float>, freqsCosSin.data_ptr<float>(), B, NH, H, D, pos.data_ptr<size_t>());

    return inp;
}

PYBIND11_MODULE(rope_paged, m)
{
    m.def("precomputeFreqsCosSin", torch::wrap_pybind_function(precomputeFreqsCosSinBind), "precomputeFreqsCosSin cpu");
    m.def("applyRope", torch::wrap_pybind_function(applyRopePagedBind), "applyRoped paged");
}