import sys
import os
import torch

# 添加编译生成的共享库文件路径
sys.path.append('../build/attentionClass')

# 确保共享库文件存在
lib_path = os.path.join(os.path.dirname(__file__), '../build/attentionClass/attentionClass.cpython-310-x86_64-linux-gnu.so')
if not os.path.exists(lib_path):
    raise ImportError(f"Cannot find shared library: {lib_path}")

# 导入生成的模块
from attentionClass import AttentionTest, LlamaConfig, DataType

# 创建 LlamaConfig 实例
config = LlamaConfig(
    32000,  # vocabSize
    4096,   # hiddenSize
    16384,  # intermediateSize
    32,     # numHiddenLayers
    32,     # numAttentionHeads
    512,    # maxPositionEmbeddings
    1,      # batch
    128,    # prefillLength
    32,     # layerNums
    10000.0, # theta
    DataType.FLOAT32,  # dataType
)

# 创建 AttentionTest 实例
model_path = "path/to/your/model.bin"  # 替换为你的模型文件路径
attention_test = AttentionTest(config, model_path)

# 调用方法
attention_test.printMessage()