import sys
import os
import time
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(os.path.dirname(__file__), '../build/matmulPaged/matmulPaged.cpython-310-x86_64-linux-gnu.so')
if not os.path.exists(lib_path):
    raise ImportError(f"Cannot find shared library: {lib_path}")
sys.path.append(os.path.join(current_dir, '../build/matmulPaged'))

from matmulPaged import matmul as matmul_paged

bsz = 1
length = 4
hidden_size = 4

input = torch.randn(bsz, length, hidden_size, dtype=torch.float32)
weight = torch.randn(hidden_size, hidden_size, dtype=torch.float32)

output_pytorch = torch.matmul(input, weight.T)
output_paged = matmul_paged(input, weight)

print(output_pytorch)
print(output_paged)

result = torch.allclose(output_paged, output_pytorch, atol=1e-6)
print("result ", result)