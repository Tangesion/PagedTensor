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
from matmulPaged import matmul_multi
from matmulPaged import matmul_block

bsz = 1
length = 1024
hidden_size = 4096

input = torch.randn(bsz, length, hidden_size, dtype=torch.float32)
weight = torch.randn(hidden_size, hidden_size, dtype=torch.float32)

output_pytorch = torch.matmul(input, weight.T)
#output_paged = matmul_paged(input, weight)
#output_paged_multi = matmul_multi(input, weight)
output_paged_block = matmul_block(input, weight)


print(output_pytorch)
#print(output_paged)
#print(output_paged_multi)
print(output_paged_block)

#result = torch.allclose(output_paged, output_pytorch, atol=1e-3)
#print("result ", result)
#
#result = torch.allclose(output_paged_multi, output_pytorch, atol=1e-3)
#print("result ", result)

result = torch.allclose(output_paged_block, output_pytorch, atol=1e-3)
print("result ", result)