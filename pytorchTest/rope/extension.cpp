#include <torch/extension.h>
#include "kernel/cpu/rope.h"

torch::Tensor precomputeFreqsCosSinBind(const long dim, const long maxPos, const float theta = 10000.0)
{
    auto freqsCosSin = torch::zeros({maxPos, 2, dim / 2}, torch::kFloat);
    inference_frame::kernel::cpu::precomputeFreqsCosSinMultiThread(freqsCosSin.data_ptr<float>(), dim, maxPos, theta);
    return freqsCosSin;
}

torch::Tensor applyRopeMultiThreadBind(torch::Tensor inp, torch::Tensor freqsCosSin, torch::Tensor pos)
{
    auto B = inp.size(0);
    auto NH = inp.size(2);
    auto H = inp.size(1);
    auto D = inp.size(3);

    // torch::Tensor out = torch::zeros_like(inp);
    inference_frame::kernel::cpu::applyRopeMultiThread(
        inp.data_ptr<float>(), freqsCosSin.data_ptr<float>(), B, NH, H, D, pos.data_ptr<size_t>());
    return inp;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("precomputeFreqsCosSin", torch::wrap_pybind_function(precomputeFreqsCosSinBind), "precomputeFreqsCosSin cpu");
    m.def("applyRope", torch::wrap_pybind_function(applyRopeMultiThreadBind), "applyRopeMultiThread cpu");
}