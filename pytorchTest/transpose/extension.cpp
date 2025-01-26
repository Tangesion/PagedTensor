#include <torch/extension.h>
#include "kernel/cpu/transpose.h"

torch::Tensor transposeBind(torch::Tensor inp)
{
    auto B = inp.size(0);
    auto H = inp.size(1);
    auto NH = inp.size(2);
    auto D = inp.size(3);

    torch::Tensor out = torch::zeros({B, NH, H, D}, torch::kFloat);
    toy::kernel::cpu::transpose(
        inp.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        NH,
        H,
        D);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", torch::wrap_pybind_function(transposeBind), "transpose cpu");
}
