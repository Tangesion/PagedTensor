import sys
import os
import torch
from transformers import LlamaForCausalLM, LlamaConfig as LlamaConfigTorch
from model import LlamaAttention as LlamaAttentionTorch, get_cos_sin

#sys.path.append('../build/attentionClass')
current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(os.path.dirname(__file__), '../build/attentionClass/attentionClass.cpython-310-x86_64-linux-gnu.so')
if not os.path.exists(lib_path):
    raise ImportError(f"Cannot find shared library: {lib_path}")
sys.path.append(os.path.join(current_dir, '../build/attentionClass'))
print(f"sys.path: {sys.path}")
from attentionClass import AttentionTest, LlamaConfig, DataType

config = LlamaConfig(
    32000,  # vocabSize
    4096,   # hiddenSize
    11008,  # intermediateSize
    32,     # numHiddenLayers
    32,     # numAttentionHeads
    512,    # maxPositionEmbeddings
    1,      # batch
    128,    # prefillLength
    32,     # layerNums
    10000.0, # theta
    DataType.FLOAT32,  # dataType
)

config_torch = LlamaConfigTorch()
config_torch.max_position_embeddings = 4096
heads_num = 32
length = 1024
bsz = 1
hidden_size = 4096
position_ids = torch.arange(length).unsqueeze(0)
inv_list = []
for i in range(0, 64):
    inv_list.append(1.0 / (10000.0 ** (2 * i / 128)))
inv_manual = torch.tensor(inv_list)
hidden_states = torch.randn(bsz, length, hidden_size)
position_embeddings = get_cos_sin(hidden_size, position_ids, inv_manual)

attention_pytorch = LlamaAttentionTorch(config_torch, 0)

model_path_golden = "/home/tgx/projects/Toy/weight/test_weights_pytorch.bin" 
model_path_test = "/home/tgx/projects/Toy/weight/test_weights_numpy.bin"
attention_test = AttentionTest(config, model_path_test)

# test if binded
attention_test.printMessage()

# test weight
weight_golden = torch.load(model_path_golden, weights_only=True)

attention_state_dict = {"q_proj.weight": weight_golden['model.layers.0.self_attn.q_proj.weight'],
                        "k_proj.weight": weight_golden['model.layers.0.self_attn.k_proj.weight'],
                        "v_proj.weight": weight_golden['model.layers.0.self_attn.v_proj.weight'],
                        "o_proj.weight": weight_golden['model.layers.0.self_attn.o_proj.weight']}
attention_pytorch.load_state_dict(attention_state_dict)

mask = torch.triu(torch.ones(bsz, heads_num, length, length), diagonal=1).bool()
causal_mask = torch.zeros(bsz, heads_num, length, length)
causal_mask.masked_fill_(mask, float('-inf'))

output_golden, _ = attention_pytorch(hidden_states, position_embeddings, causal_mask)

q_proj_golden = weight_golden['model.layers.0.self_attn.q_proj.weight'].to(torch.float32)

q_proj_test = attention_test.getQProj(config)
result = torch.allclose(q_proj_golden, q_proj_test, atol=1e-6)
print("q_proj test result: ", result)


k_proj_golden = weight_golden['model.layers.0.self_attn.k_proj.weight'].to(torch.float32)
k_proj_test = attention_test.getKProj(config)
result = torch.allclose(k_proj_golden, k_proj_test, atol=1e-6)
print("k_proj test result: ", result)

v_proj_golden = weight_golden['model.layers.0.self_attn.v_proj.weight'].to(torch.float32)
v_proj_test = attention_test.getVProj(config)
result = torch.allclose(v_proj_golden, v_proj_test, atol=1e-6)
print("v_proj test result: ", result)

out_proj_golden = weight_golden['model.layers.0.self_attn.o_proj.weight'].to(torch.float32)
out_proj_test = attention_test.getOProj(config)
result = torch.allclose(out_proj_golden, out_proj_test, atol=1e-6)
print("o_proj test result: ", result)


