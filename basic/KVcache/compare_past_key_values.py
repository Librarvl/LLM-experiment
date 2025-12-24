import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_past_key_values():
    model = AutoModelForCausalLM.from_pretrained("/home/boran.lbr/LLM_model/gpt2")
    tokenizer = AutoTokenizer.from_pretrained("/home/boran.lbr/LLM_model/gpt2")

    prompt = "Hello, my name is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # 使用 KV cache 生成
    past_key_values = None
    generated = input_ids

    for _ in range(20):  # 生成 20 个 token
        outputs = model(
            input_ids=generated if past_key_values is None else generated[:, -1:],
            past_key_values=past_key_values,
            use_cache=True
        )
        
        past_key_values = outputs.past_key_values  # 更新缓存
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    print(tokenizer.decode(generated[0]))
    return model, input_ids


def compare_using_time(model, input_ids):

    # 不使用 cache
    start = time.time()
    output = model.generate(input_ids, max_length=100, use_cache=False)
    print(f"Without cache: {time.time() - start:.2f}s")

    # 使用 cache
    start = time.time()
    output = model.generate(input_ids, max_length=100, use_cache=True)
    print(f"With cache: {time.time() - start:.2f}s")

    # 结果示例:
    # Without cache: 5.32s
    # With cache: 1.28s  # 快 4 倍！


def calc_using_memory():
    # 内存计算
    batch_size = 1
    num_layers = 32
    num_heads = 32
    head_dim = 128
    seq_len = 2048

    memory_per_layer = 2 * batch_size * num_heads * seq_len * head_dim * 2  # 2 bytes (fp16)
    total_memory = memory_per_layer * num_layers / (1024**3)  # GB

    print(f"KV Cache 内存占用: {total_memory:.2f} GB")
    # 输出: KV Cache 内存占用: 1.05 GB


def main():
    model, input_ids = apply_past_key_values()
    compare_using_time(model, input_ids)
    # calc_using_memory()       # 没理解


if __name__ == "__main__":
    main()