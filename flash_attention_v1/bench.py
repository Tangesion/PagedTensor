import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Load the CUDA kernel as a python module
minimal_attn = load(name='minimal_attn', sources=['main.cpp', 'flash_attn.cu'], extra_cuda_cflags=['-O2'])

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 16
n_head = 12
seq_len = 128
head_embd = 32

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
#torch.set_printoptions(edgeitems=1000, threshold=1000000)
#print(minimal_attn.forward(q, k, v,1024))

#"""
print('=== profiling manual attention ===')
#(1.0 / math.sqrt(k.size(-1)))
# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
torch.set_printoptions(edgeitems=1000, threshold=1000000)
def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y
#"""
#"""
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = manual_attn(q, k, v)
    print(manual_result[0,0,:,:])
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling minimal flash attention === ')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_result = minimal_attn.forward(q, k, v, 1024)
    print(minimal_result[0,0,:,:])
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
#print(minimal_result)
#"""
print('attn values sanity check:', torch.allclose(minimal_result, manual_result, rtol=0, atol=1e-02))
#minimal_result and manual_result differ num of elements
"""
num_diff = torch.sum(minimal_result == manual_result)
print(f"The number of same elements is {num_diff}, which is {num_diff / minimal_result.numel()} of the total elements.")
num_diff_1 = torch.sum(minimal_result[0,0,:,:] == manual_result[0,0,:,:])
print(f"The number of same elements is {num_diff_1}, which is {num_diff_1 / minimal_result[0,:,:,:].numel()} of the total elements.")
num_diff_2 = torch.sum(minimal_result[0,2,:,:] == manual_result[0,2,:,:])
print(f"The number of same elements is {num_diff_2}, which is {num_diff_2 / minimal_result[1,:,:,:].numel()} of the total elements.")
"""

#"""
#torch.set_printoptions(edgeitems=1000, threshold=1000000)

"""

"""
"""
import numpy as np
score_std = q @ k.transpose(-2, -1)/math.sqrt(k.size(-1))
#print('score_std:', score_std.shape)
#print(score_std[0,1,32:64,:])
score_flash = minimal_attn.forward(q, k, v, 1024)
#print('score_flash:', score_flash.shape)
#print(score_flash[0,1,32:64,:])
matrix1 = score_std[0,1,32:64,:].cpu().numpy()
matrix2 = score_flash[0,1,32:64,:].cpu().numpy()
for i in range(matrix1.shape[0]):
    if not np.all(matrix1[i] == matrix2[i]):
        print(f"The matrices start to differ from row {i}")
        break
matrix1 = score_std[0,1,32+22,:].cpu().numpy()
matrix2 = score_flash[0,1,32+22,:].cpu().numpy()
start_col = None
end_col = None
for i in range(matrix1.shape[0]):
    if not np.all(matrix1[i] == matrix2[i]):
        if start_col is None:
            start_col = i
        end_col = i

if start_col is not None:
    print(f"The matrices start to differ from column {start_col} and end at column {end_col}")

matrix1 = score_std
matrix2 = score_flash
num_diff = torch.sum(matrix1 != matrix2)

# Calculate the total number of elements
total_elements = matrix1.numel()

# Calculate the proportion of different elements
proportion_diff = num_diff / total_elements

print(f"The proportion of different elements is {proportion_diff}")
print(f"The number of different elements is {num_diff}")

print('attn values sanity check:', torch.allclose(score_std, score_flash, rtol=0, atol=1e-02))
#print(score_flash[0,1,32 + 22 :64,:])
#print(score_std[0,1,32 + 22 :64,:])
#"""

#torch.set_printoptions(edgeitems=1000, threshold=1000000)
#out_k = minimal_attn.forward(q, k, v, 1024)
#import numpy as np

# Assuming out_k is a PyTorch tensor, convert it to a numpy array first
#out_k_np = out_k[0, 0, :, :].cpu().numpy()

# Find the index of the first row that all elements are 0
#zero_row_index = np.argmax(np.all(out_k_np == 0, axis=1))

#print('The first all-zero row is at index:', zero_row_index)

#print(out_q)

#print(q)
#print('attn values sanity check:', torch.allclose(out_k, k, rtol=0, atol=1e-02))
