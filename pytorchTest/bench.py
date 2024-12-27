import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import math

cpu_attn = load(name='cpu_attn', sources=['extension.cpp'], extra_cflags=['-O3'], extra_include_paths=['/home/gexingt/tgx/projects/inference-frame/cpp/include'])

batch_size = 1
n_head = 1
seq_len = 4
head_embd = 4

q = torch.randn(batch_size, n_head, seq_len, head_embd, dtype=torch.float32)
k = torch.randn(batch_size, n_head, seq_len, head_embd, dtype=torch.float32)
v = torch.randn(batch_size, n_head, seq_len, head_embd, dtype=torch.float32)

def manual_attn(q, k, v):
    att = q @ k.transpose(-2, -1) / math.sqrt(head_embd)
    print(att)
    # Create an upper triangular matrix with -inf on the upper triangle and 0 on the lower triangle
    mask = torch.triu(torch.ones(att.size()), diagonal=1).bool()
    att.masked_fill_(mask, float('-inf'))
    
    # Subtract max for numerical stability
    att = att - att.max(dim=-1, keepdim=True)[0]
    att = F.softmax(att, dim=-1)
    
    y = att @ v
    return y



print('=== profiling pytorch manual attention ===')
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = manual_attn(q, k, v)
    #print(manual_result[0,0,:,:])
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling cpu attention === ')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    cpu_attn_result, score = cpu_attn.forward(q, k, v)
    #print(minimal_result[0,0,:,:])
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
print(score)
#print(minimal_result)
#"""
#print(manual_result)
#rint(cpu_attn_result)
print('attn values sanity check:', torch.allclose(cpu_attn_result, manual_result, rtol=0, atol=1e-02))