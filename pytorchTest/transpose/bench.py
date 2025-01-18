import torch
import os
from torch.utils.cpp_extension import load

current_dir = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(current_dir, 'extension.cpp')
include_path = os.path.join(current_dir, '../../cpp/include')

cpu_transpose = load(name='cpu_transpose', sources=[source_path], extra_cflags=['-O3'], extra_include_paths=[include_path])

#cpu_transpose = load(name='cpu_transpose', sources=['extension.cpp'], extra_cflags=['-O3'], extra_include_paths=['/home/gexingt/tgx/projects/inference-frame/cpp/include'])

batch_size = 1
seq_len = 1024
head_dim = 128
head_num = 32

x = torch.randn(batch_size, seq_len, head_num, head_dim, dtype=torch.float32)

x_trans = x.transpose(1, 2)
print(x_trans.shape)

result = cpu_transpose.forward(x)

print('transpose values sanity check:', torch.allclose(result, x_trans, rtol=0, atol=1e-05))