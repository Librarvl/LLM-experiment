from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

# 2D RoPE 可能存在于视觉模型中
# 例如：transformers.models.*.modeling_*.py

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('running_mean', torch.zeros(10))
        self.weight = nn.Parameter(torch.randn(10))  # 对比：需要梯度


def main():
    # LlamaRotaryEmbedding()      # 1D RoPE
    MyModel()


if __name__ == "__main__":
    main()