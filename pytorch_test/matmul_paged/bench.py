import sys
import os
import time
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(os.path.dirname(__file__), '../build/matmul_paged/matmul_paged.cpython-310-x86_64-linux-gnu.so')
if not os.path.exists(lib_path):
    raise ImportError(f"Cannot find shared library: {lib_path}")
sys.path.append(os.path.join(current_dir, '../build/matmul_paged'))

from matmul_paged import matmul as matmul_paged
from matmul_paged import matmul_multi
from matmul_paged import matmul_block
from matmul_paged import matmul_block_multi

bsz = 1
length = 32
hidden_size = 4096

input = torch.randn(bsz, length, hidden_size, dtype=torch.float32)
weight = torch.randn(hidden_size, hidden_size, dtype=torch.float32)

output_pytorch = torch.matmul(input, weight.T)
#output_paged = matmul_paged(input, weight)
#output_paged_multi = matmul_multi(input, weight)
output_paged_block = matmul_block(input, weight)
output_paged_block_multi = matmul_block_multi(input, weight)

#print(output_pytorch)
#print(output_paged)
#print(output_paged_multi)
#print(output_paged_block)
#print(output_paged_block_multi)

#result = torch.allclose(output_paged, output_pytorch, atol=1e-3)
#print("result ", result)
#
#result = torch.allclose(output_paged_multi, output_pytorch, atol=1e-3)
#print("result ", result)

result = torch.allclose(output_paged_block, output_pytorch, atol=1e-3)
print("result ", result)

result = torch.allclose(output_paged_block_multi, output_pytorch, atol=1e-3)
print("result ", result)
# 找出不匹配的元素
diff_mask = ~torch.isclose(output_paged_block_multi, output_pytorch, atol=1e-1)

## 打印不匹配的元素及其索引
print("Indices of differing elements:")
print(diff_mask.nonzero(as_tuple=True))

print("Values in output_pytorch:")
print(output_pytorch[diff_mask])

print("Values in output_paged_block:")
print(output_paged_block_multi[diff_mask])