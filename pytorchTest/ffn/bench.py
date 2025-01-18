import torch
import math
import os
from torch.utils.cpp_extension import load

current_dir = os.path.dirname(os.path.abspath(__file__))
extension_path = os.path.join(current_dir, 'extension.cpp')
include_path = os.path.join(current_dir, '../../cpp/include')

cpu_ffn = load(name='cpu_ffn', sources=[extension_path], extra_cflags=['-O3'], extra_include_paths=[include_path])

#cpu_ffn = load(name='cpu_ffn', sources=['extension.cpp'], extra_cflags=['-O3'], extra_include_paths=['/home/gexingt/tgx/projects/inference-frame/cpp/include'])

batch_size = 1
seq_len = 16
dim = 32
intern_dim = 64
x = torch.randn(batch_size, seq_len, dim, dtype=torch.float32)
gate_proj = torch.randn(intern_dim, dim, dtype=torch.float32)
up_proj = torch.randn(intern_dim, dim, dtype=torch.float32)
down_proj = torch.randn(dim, intern_dim, dtype=torch.float32)

def pytorch_ffn(x, gate_proj, up_proj, down_proj):
    #silu ffn
    up_proj_out = x @ up_proj.t()
    gate_proj_out = x @ gate_proj.t()
    gate_proj_out = gate_proj_out * torch.sigmoid(gate_proj_out)
    gate_proj_out = up_proj_out * gate_proj_out
    down_proj_out = gate_proj_out @ down_proj.t()
    return down_proj_out

result_py = pytorch_ffn(x, gate_proj, up_proj, down_proj)
#result_cpu = cpu_ffn.ffnForwardOneThread(x, gate_proj, up_proj, down_proj)
result_cpu = cpu_ffn.ffnForwardMultiThread(x, gate_proj, up_proj, down_proj)
print(result_py)
print(result_cpu)

#check the result
print('ffn values sanity check:', torch.allclose(result_cpu, result_py, rtol=0, atol=1e-02))

