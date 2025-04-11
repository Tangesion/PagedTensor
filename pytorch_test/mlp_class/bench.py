import sys
import os
import time
import torch
from transformers import LlamaConfig as LlamaConfigTorch
from transformers.models.llama.modeling_llama import LlamaMLP as LlamaMLPTorch

current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(os.path.dirname(__file__), '../build/mlp_class/mlp_class.cpython-310-x86_64-linux-gnu.so')
if not os.path.exists(lib_path):
    raise ImportError(f"Cannot find shared library: {lib_path}")
sys.path.append(os.path.join(current_dir, '../build/mlp_class'))
print(f"sys.path: {sys.path}")

from mlp_class import MLPTest, LlamaConfig, DataType

config = LlamaConfig(
    32000,  # vocabSize
    4096,   # hiddenSize
    11008,  # intermediateSize
    32,     # numHiddenLayers
    32,     # numAttentionHeads
    4096,    # maxPositionEmbeddings
    32,     # layerNums
    10000.0, # theta
    DataType.FLOAT32,  # dataType
)

config_torch = LlamaConfigTorch()
config_torch.max_position_embeddings = 4096

bsz = 1
length = 1000
hidden_size = 4096

mlp_pytorch = LlamaMLPTorch(config_torch)

model_path_golden = os.path.abspath(os.path.join(current_dir, "../../weight/test_weights_pytorch.bin"))
model_path_test = os.path.abspath(os.path.join(current_dir, "../../weight/test_weights_numpy.bin"))

mlp_test = MLPTest(config, model_path_test, 0)
mlp_test.printMessage()

weight_golden = torch.load(model_path_golden, weights_only=True)
#print(weight_golden.keys())
mlp_state_dict = {
    "gate_proj.weight": weight_golden["model.layers.0.mlp.gate_proj.weight"],
    "up_proj.weight": weight_golden["model.layers.0.mlp.up_proj.weight"],
    "down_proj.weight": weight_golden["model.layers.0.mlp.down_proj.weight"]
}
mlp_pytorch.load_state_dict(mlp_state_dict)

gate_proj_golden = weight_golden["model.layers.0.mlp.gate_proj.weight"].to(torch.float32)
up_proj_golden = weight_golden["model.layers.0.mlp.up_proj.weight"].to(torch.float32)
down_proj_golden = weight_golden["model.layers.0.mlp.down_proj.weight"].to(torch.float32)

gate_proj_test = mlp_test.getGateProj(config)
up_proj_test = mlp_test.getUpProj(config)
down_proj_test = mlp_test.getDownProj(config)

#print(gate_proj_test.shape)
#print(gate_proj_test)

result = torch.allclose(gate_proj_golden, gate_proj_test, atol=1e-6)
print("gate_proj test result: ", result)
result = torch.allclose(up_proj_golden, up_proj_test, atol=1e-6)
print("up_proj test result: ", result)
result = torch.allclose(down_proj_golden, down_proj_test, atol=1e-6)
print("down_proj test result: ", result)

hidden_states = torch.randn(bsz, length, hidden_size, dtype=torch.float32)

start_time = time.time()
output_golden= mlp_pytorch(hidden_states)
end_time = time.time()
print(f"pytorch prefill time: {end_time - start_time}")

output_test = torch.zeros_like(output_golden)
start_time = time.time()
output = mlp_test.forwardTest(output_test, hidden_states)
end_time = time.time()
print(f"paged_tensor prefill time: {end_time - start_time}")

result = torch.allclose(output_golden, output, atol=1e-4)
print("output_prefill test result: ", result)

# decode
hidden_states_decode = torch.randn(bsz, 1, hidden_size, dtype=torch.float32)
# Decode phase
start_time = time.time()
output_golden_decode = mlp_pytorch(hidden_states_decode)
end_time = time.time()
print(f"pytorch decode time: {end_time - start_time}")

output_test_decode = torch.zeros_like(output_golden_decode)
start_time = time.time()
output_decode = mlp_test.forwardTest(output_test_decode, hidden_states_decode)
end_time = time.time()
print(f"paged_tensor decode time: {end_time - start_time}")

# Compare decode outputs
result_decode = torch.allclose(output_golden_decode, output_decode, atol=1e-4)
print("output_decode test result: ", result_decode)