import torch
from torch.utils.cpp_extension import load



def get_cos_sin(x, position_ids, inv_freq):
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin

def get_cos_sin_half(x, position_ids, inv_freq):
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = freqs
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

cpu_rope = load(name='cpu_rope', sources=['extension.cpp'], extra_cflags=['-O3'], extra_include_paths=['/home/gexingt/tgx/projects/inference-frame/cpp/include'])
dim = 128
max_len = 4096
theta = 10000
result = cpu_rope.precomputeFreqsCosSin(dim, max_len, theta)
#print(result.shape)
#print(result)
#print(result.shape)

q = torch.randn(1, 1, 32, 128, dtype=torch.float32)
q_clone = q.clone()
position = torch.arange(1024, 1025).to(torch.uint64)
result_rope = cpu_rope.applyRope(q, result, position)
print(result_rope)
#print(result.shape)

#model_weight = torch.load("/home/gexingt/tgx/models/Llama-2-7b-hf/pytorch_model-00001-of-00002.bin")
#inv_freq = model_weight['model.layers.0.self_attn.rotary_emb.inv_freq']
inv_list = []
for i in range(0, 64):
    inv_list.append(1.0 / (10000.0 ** (2 * i / 128)))
inv_manual = torch.tensor(inv_list)
#q = torch.randn(1, 32, 1024, 128)
position = torch.arange(1024, 1025).unsqueeze(0)
cos, sin = get_cos_sin(q, position, inv_manual)
#print(cos)
#print(sin)
q_embed = apply_rotary_pos_emb(q_clone, cos, sin, position, unsqueeze_dim=2)
print(q_embed)
#check
print('rope values sanity check:', torch.allclose(q_embed, result_rope, rtol=0, atol=1e-03))
#print(result)
#position = torch.arange(1024).unsqueeze(0)
#cos_half, sin_half = get_cos_sin_half(q, position, inv_manual)
#con_sin_half = torch.cat([cos_half, sin_half], dim=-1).view(4096, 2, 64)
#print(con_sin_half)
##check
#print('rope values sanity check:', torch.allclose(con_sin_half, result, rtol=0, atol=1e-03))