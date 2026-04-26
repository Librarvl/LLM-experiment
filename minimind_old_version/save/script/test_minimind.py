import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM


import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


# ============ 1. 加载模型和tokenizer ============
# model_path = "/home/boran.lbr/LLM_model/MiniMind2"  # 替换成你的模型路径
# tokenizer_path = "/home/boran.lbr/LLM_model/MiniMind2"

model_path = "/home/boran.lbr/gitspace/github/minimind/save/hf/sft"  # 替换成你的模型路径
tokenizer_path = "/home/boran.lbr/gitspace/github/minimind/model"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = MiniMindForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,   # 用bfloat16节省显存
    # torch_dtype=torch.float32,   # 用bfloat16节省显存
    # device_map="auto",            # 自动分配GPU/CPU
    device_map="cuda:1",            # 自动分配GPU/CPU
)
model.eval()  # 推理模式，关闭dropout


def _create_chat_prompt(tokenizer, cs):
    messages = cs.copy()
    tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
    )


# ============ 2. 基础生成函数 ============
def generate(
    prompt,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
):
    messages = [
        {"role": "user", "content": "你好"}
    ]

    inputs = _create_chat_prompt(tokenizer, messages)
    inputs = tokenizer(inputs, return_tensors="pt").to(model.device)
    inputs.pop("token_type_ids", None)

    input_len = inputs["input_ids"].shape[1]

    # 2.2 生成
    with torch.no_grad():                     # 不计算梯度
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,    # 最多生成多少token
            do_sample=do_sample,              # True=采样, False=贪心
            temperature=temperature,           # 温度，越高越随机
            top_p=top_p,                      # 核采样
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 2.3 只取新生成的部分（去掉输入）
    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return response




# ============ 5. 生成参数说明与效果 ============
"""
temperature（温度）：
  0.1  → 非常保守，几乎总选最高概率词，重复性高
  0.7  → 平衡，推荐值
  1.0  → 原始分布，较随机
  1.5  → 非常随机，可能乱说

top_p（核采样）：
  0.1  → 只从概率最高的10%词中采样，保守
  0.9  → 从累计概率90%的词中采样，推荐值
  1.0  → 不过滤，全部词都可能被选到

top_k（Top-K采样）：
  只从概率最高的 k 个词中采样
  top_k=50  → 每次只考虑前50个词

do_sample：
  True  → 采样模式（有随机性）
  False → 贪心模式（总选最高概率，确定性输出）
  
repetition_penalty（重复惩罚）：
  1.0  → 不惩罚
  1.3  → 轻度惩罚重复，推荐
  2.0  → 强烈惩罚，可能影响质量
"""

# ============ 6. 实际调用 ============
if __name__ == "__main__":
    
    # 基础生成
    print("=== 基础生成 ===")
    prompt = "请介绍一下人工智能的发展历史"
    response = generate(prompt, max_new_tokens=300, temperature=0.7)
    print(f"输入: {prompt}")
    print(f"输出: {response}")
    print()
    
    # # 流式生成
    # print("=== 流式生成（逐字输出）===")
    # generate_stream("写一首关于春天的诗")
    # print()
    
    # # 多轮对话
    # print("=== 多轮对话 ===")
    # history = []
    # system = "你是一个helpful的AI助手"
    
    # questions = [
    #     "你好，你是谁？",
    #     "你能做什么？",
    #     "帮我写一段Python代码",
    # ]
    
    # for q in questions:
    #     print(f"用户: {q}")
    #     response, history = chat(q, history, system_prompt=system)
    #     print(f"助手: {response}")
    #     print("-" * 40)
