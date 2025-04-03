import torch
from tqdm import tqdm
import os

llama2_path = os.environ.get("LLAMA2_PATH")

model_weight1_path = os.path.join(llama2_path, "pytorch_model-00001-of-00002.bin")
model_weight2_path = os.path.join(llama2_path, "pytorch_model-00002-of-00002.bin")

# 加载模型权重
model_weight1 = torch.load(model_weight1_path, weights_only=True, map_location='cpu')
model_weight2 = torch.load(model_weight2_path, weights_only=True, map_location='cpu')
with open("weight/model_weight.bin", "wb") as f:
    for key in tqdm(model_weight1.keys()):
        weight = model_weight1[key].to(torch.float32).numpy()
        weight.tofile(f)
    for key in tqdm(model_weight2.keys()):
        weight = model_weight2[key].to(torch.float32).numpy()
        weight.tofile(f)

model_weight_pytorch = torch.load("/home/tgx/projects/paged_tensor/weight/test_weights_pytorch.bin")
with open("/home/tgx/projects/paged_tensor/weight/test_weights_numpy.bin", "wb") as f:
    for key in model_weight_pytorch.keys():
        weight = model_weight_pytorch[key].to(torch.float32).numpy()
        weight.tofile(f)