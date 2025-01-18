import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import math
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
extension_path = os.path.join(current_dir, 'extension.cpp')
include_path = os.path.join(current_dir, '../../cpp/include')

cpu_attn = load(name='cpu_attn', sources=[extension_path], extra_cflags=['-O3'], extra_include_paths=[include_path])

batch_size = 1
n_head = 1
seq_len = 4
head_embd = 4

q = torch.randn(batch_size, n_head, seq_len, head_embd, dtype=torch.float32)
k = torch.randn(batch_size, n_head, seq_len, head_embd, dtype=torch.float32)
v = torch.randn(batch_size, n_head, seq_len, head_embd, dtype=torch.float32)

def prefill_attn(q, k, v):
    att = q @ k.transpose(-2, -1) / math.sqrt(head_embd)
    #print(att)
    # Create an upper triangular matrix with -inf on the upper triangle and 0 on the lower triangle
    mask = torch.triu(torch.ones(att.size()), diagonal=1).bool()
    att.masked_fill_(mask, float('-inf'))
    
    # Subtract max for numerical stability
    #att = att - att.max(dim=-1, keepdim=True)[0]
    att = F.softmax(att, dim=-1)
    
    y = att @ v
    return y

q_vec = torch.randn(batch_size, n_head, 1, head_embd, dtype=torch.float32)

def decode_attn(q_vec, k, v):
    att = q_vec @ k.transpose(-2, -1) / math.sqrt(head_embd)
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

prefill_result = prefill_attn(q, k, v)
decode_result = decode_attn(q_vec, k, v)

cpu_prefill_result, score = cpu_attn.prefill(q, k, v)
cpu_decode_result, score = cpu_attn.decode(q_vec, k, v)

print('prefill attn values sanity check:', torch.allclose(cpu_prefill_result, prefill_result, rtol=0, atol=1e-02))
print('decode attn values sanity check:', torch.allclose(cpu_decode_result, decode_result, rtol=0, atol=1e-02))