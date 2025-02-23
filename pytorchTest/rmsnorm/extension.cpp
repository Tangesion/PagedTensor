#include <torch/extension.h>
#include "kernel/cpu/rmsnorm.h"

torch::Tensor rmsnormBind(torch::Tensor inp, torch::Tensor weight)
{
    auto B = inp.size(0);
    auto H = inp.size(1);
    auto C = inp.size(2);
    torch::Tensor out = torch::zeros_like(inp);
    paged_tensor::kernel::cpu::rmsNormMultiThread(
        out.data_ptr<float>(),
        inp.data_ptr<float>(), weight.data_ptr<float>(),
        B, H, C);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", torch::wrap_pybind_function(rmsnormBind), "rmsnorm cpu");
}