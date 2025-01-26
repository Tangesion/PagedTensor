#include <torch/extension.h>
#include "kernel/cpu/ffn.h"

torch::Tensor ffnForwardOneThreadBind(torch::Tensor inp, torch::Tensor gateProj, torch::Tensor upProj, torch::Tensor downProj)
{
    auto B = inp.size(0);
    auto H = inp.size(1);
    auto C = inp.size(2);
    auto interSize = gateProj.size(0);

    torch::Tensor out = torch::zeros_like(inp);
    torch::Tensor outInterGate = torch::zeros({B, H, interSize}, torch::kFloat);
    torch::Tensor outInterUp = torch::zeros({B, H, interSize}, torch::kFloat);

    toy::kernel::cpu::ffnForwardOneThread(
        out.data_ptr<float>(),
        inp.data_ptr<float>(),
        outInterGate.data_ptr<float>(),
        outInterUp.data_ptr<float>(),
        gateProj.data_ptr<float>(),
        upProj.data_ptr<float>(),
        downProj.data_ptr<float>(),
        B,
        H,
        C,
        interSize);

    return out;
}

torch::Tensor ffnForwardMultiThreadBind(torch::Tensor inp, torch::Tensor gateProj, torch::Tensor upProj, torch::Tensor downProj)
{
    auto B = inp.size(0);
    auto H = inp.size(1);
    auto C = inp.size(2);
    auto interSize = gateProj.size(0);

    torch::Tensor out = torch::zeros_like(inp);
    torch::Tensor outInterGate = torch::zeros({B, H, interSize}, torch::kFloat);
    torch::Tensor outInterUp = torch::zeros({B, H, interSize}, torch::kFloat);

    toy::kernel::cpu::ffnForwardMultiThread(
        out.data_ptr<float>(),
        inp.data_ptr<float>(),
        outInterGate.data_ptr<float>(),
        outInterUp.data_ptr<float>(),
        gateProj.data_ptr<float>(),
        upProj.data_ptr<float>(),
        downProj.data_ptr<float>(),
        B,
        H,
        C,
        interSize);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ffnForwardOneThread", torch::wrap_pybind_function(ffnForwardOneThreadBind), "ffnForwardOneThread cpu");
    m.def("ffnForwardMultiThread", torch::wrap_pybind_function(ffnForwardMultiThreadBind), "ffnForwardMultiThread cpu");
}