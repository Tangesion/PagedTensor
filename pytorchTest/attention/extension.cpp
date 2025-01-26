#include <torch/extension.h>
#include "kernel/cpu/attention.h"
std::tuple<torch::Tensor, torch::Tensor> attentionPrefillBind(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    auto B = q.size(0);
    auto NH = q.size(1);
    auto H = q.size(2);
    auto D = q.size(3);

    torch::Tensor out = torch::zeros_like(q);
    torch::Tensor interAttn = torch::zeros({B, NH, H, H});
    toy::kernel::cpu::attentionForwardMultiThread(
        out.data_ptr<float>(),
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        interAttn.data_ptr<float>(),
        true,
        B, NH, H, D);
    return std::make_tuple(out, interAttn);
}

std::tuple<torch::Tensor, torch::Tensor> attentiondecodeBind(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    auto B = q.size(0);
    auto NH = q.size(1);
    auto H = k.size(2);
    auto D = q.size(3);

    torch::Tensor out = torch::zeros_like(q);
    torch::Tensor interAttn = torch::zeros({B, NH, 1, H});
    toy::kernel::cpu::attentionForwardMultiThread(
        out.data_ptr<float>(),
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        interAttn.data_ptr<float>(),
        false,
        B, NH, H, D);
    return std::make_tuple(out, interAttn);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("prefill", torch::wrap_pybind_function(attentionPrefillBind), "attention cpu prefill");
    m.def("decode", torch::wrap_pybind_function(attentiondecodeBind), "attention cpu decode");
}
