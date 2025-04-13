import sys
import os
import time
import torch
from torch.nn import functional as F
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(os.path.dirname(__file__), '../build/attention_paged/attention_paged.cpython-310-x86_64-linux-gnu.so')
if not os.path.exists(lib_path):
    raise ImportError(f"Cannot find shared library: {lib_path}")
sys.path.append(os.path.join(current_dir, '../build/attention_paged'))

from attention_paged import forward_decode, forward_prefill

batch_size = 1
n_head = 32
seq_len = 1024
head_embd = 128

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

paged_prefill_result = forward_prefill(q, k, v)
paged_decode_result = forward_decode(q_vec, k, v)

print('prefill attn values sanity check:', torch.allclose(paged_prefill_result, prefill_result, rtol=0, atol=1e-03))
print('decode attn values sanity check:', torch.allclose(paged_decode_result, decode_result, rtol=0, atol=1e-03))