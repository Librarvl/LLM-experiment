# convert_to_hf.py
import torch
from transformers import AutoTokenizer

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

def convert_pth_to_hf(
    pth_path,           # 原始 .pth 文件路径
    save_dir,           # 保存 HF 格式的目录
    tokenizer_path,     # tokenizer 路径
):
    print(f"加载 .pth 权重: {pth_path}")
    
    # 1. 加载原始权重
    state_dict = torch.load(pth_path, map_location="cpu")
    
    # 如果是完整checkpoint，取出model部分
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    
    # 2. 创建 HF 格式的配置
    # config = MiniMindConfig(
    #     hidden_size=768,
    #     num_layers=16,
    #     num_heads=8,
    #     vocab_size=6400,
    #     max_seq_len=1024,
    # )
    
    # 3. 创建 HF 格式的模型
    # model = MiniMindForCausalLM(config)
    model = MiniMindForCausalLM(MiniMindConfig(
        hidden_size=768,
        num_hidden_layers=16,
        use_moe=False,
        inference_rope_scaling=False
    ))
    
    # 4. 加载权重（可能需要处理key的映射）
    # 检查key是否匹配
    hf_keys = set(model.state_dict().keys())
    pth_keys = set(state_dict.keys())
    
    missing  = hf_keys - pth_keys
    unexpected = pth_keys - hf_keys
    
    if missing:
        print(f"⚠️  缺少的key: {missing}")
    if unexpected:
        print(f"⚠️  多余的key: {unexpected}")
    
    model.load_state_dict(state_dict, strict=False)
    print("✅ 权重加载成功")
    
    # 5. 保存为 HF 格式
    import copy
    import torch.nn as nn

    model_to_save = copy.deepcopy(model)
    # 解除权重绑定，给 lm_head 独立的权重
    model_to_save.lm_head.weight = nn.Parameter(
        model.lm_head.weight.clone()  # clone 出独立副本
    )
    model_to_save.save_pretrained(save_dir)

    # model.save_pretrained(save_dir)   # 保存 model.safetensors + config.json
    print(f"✅ 模型保存到: {save_dir}")
    
    # 6. 保存 tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # tokenizer.save_pretrained(save_dir)
    # print(f"✅ Tokenizer 保存到: {save_dir}")
    
    # 7. 验证可以正常加载
    print("验证加载...")
    loaded_model = MiniMindForCausalLM.from_pretrained(save_dir)
    print(f"✅ 验证成功！模型参数量: {sum(p.numel() for p in loaded_model.parameters()):,}")


if __name__ == "__main__":
    convert_pth_to_hf(
        pth_path="E:\\code\\minimind_old\\save\\full_sft_data512_10ep_len512_768.pth",
        save_dir="E:\\code\\minimind_old\\save\\hf\\SFT_v0",
        tokenizer_path="E:\\code\\minimind_old\\save\\hf\\SFT_v0",
    )
