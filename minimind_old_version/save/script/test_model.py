import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM


import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


# ============ 1. 加载模型和tokenizer ============
model_path = "/home/boran.lbr/gitspace/github/minimind/save/hf/mini_sft_v1"  # 替换成你的模型路径
tokenizer_path = "/home/boran.lbr/gitspace/github/minimind/model"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = MiniMindForCausalLM.from_pretrained(
    model_path,
    # torch_dtype=torch.bfloat16,   # 用bfloat16节省显存
    torch_dtype=torch.float32,   # 用bfloat16节省显存
    # device_map="auto",            # 自动分配GPU/CPU
    device_map="cuda:1",            # 自动分配GPU/CPU
)
model.eval()  # 推理模式，关闭dropout

model = MiniMindForCausalLM(MiniMindConfig(
    hidden_size=768,
    num_hidden_layers=16,
    use_moe=False,
    inference_rope_scaling=False
))
ckp = f'/home/boran.lbr/gitspace/github/minimind/save/pretrain_1st_4ep_h768_bs128_768.pth'
model.load_state_dict(torch.load(ckp, map_location="cuda:0"), strict=True)
model.eval().to("cuda:0")


def _create_chat_prompt(tokenizer, cs):
    messages = cs.copy()
    tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools
    )


# ============ 2. 基础生成函数 ============
def generate(
    prompt,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
):
    # 2.1 tokenize 输入
    # inputs = tokenizer(
    #     prompt,
    #     return_tensors="pt",      # 返回PyTorch tensor
    # ).to(model.device)
    # inputs.pop("token_type_ids", None)

    # input_len = inputs["input_ids"].shape[1]  # 记录输入长度
    # print(input_len)

    messages = [
        {"role": "user", "content": "写一首关于春天的诗"}
        # {"role": "user", "content": "你好"}
    ]
    prompt_with_template = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        # return_tensors="pt",         # 返回 tensor
    )
    print(prompt_with_template)
    # print(inputs["input_ids"].shape[1])

    inputs = tokenizer(
        prompt_with_template,
        return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    print(inputs)
    input_len = inputs["input_ids"].shape[1]  # 记录输入长度
    print(input_len)

    texts = tokenizer.batch_decode(
        inputs.input_ids.tolist(),          # tensor → list of list
        skip_special_tokens=True,
    )
    print(texts)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generated_ids = model.generate(
        inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
        max_new_tokens=512, do_sample=True, streamer=streamer,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        top_p=0.85, temperature=0.85, repetition_penalty=1.0
    )
    response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

    print("*" * 80)

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
    print(response)


    return response


# ============ 3. 流式生成（逐字输出）============
from transformers import TextStreamer

def generate_stream(prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs.pop("token_type_ids", None)
    
    # TextStreamer 会实时打印生成的token
    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,          # 不打印输入部分
        skip_special_tokens=True,  # 不打印特殊token
    )
    
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            streamer=streamer,          # ← 关键：传入streamer
            temperature=0.7,
            do_sample=True,
        )


# ============ 4. 对话格式生成 ============
def chat(
    user_input,
    history=None,       # 历史对话列表
    system_prompt=None, # 系统提示词
):
    if history is None:
        history = []
    
    # 4.1 构建对话格式（不同模型格式不同）
    # ChatML 格式（很多模型用这个）
    messages = []
    
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    # 加入历史对话
    for user_msg, assistant_msg in history:
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    # 加入当前输入
    messages.append({"role": "user", "content": user_input})
    
    # 4.2 用 apply_chat_template 转成模型输入格式
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,   # 加上 <|assistant|> 之类的前缀
    )
    
    # 4.3 生成回复
    response = generate(prompt)
    
    # 4.4 更新历史
    history.append((user_input, response))
    
    return response, history


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
    
    # 流式生成
    print("=== 流式生成（逐字输出）===")
    generate_stream("写一首关于春天的诗")
    print()
    
    # 多轮对话
    print("=== 多轮对话 ===")
    history = []
    system = "你是一个helpful的AI助手"
    
    questions = [
        "你好，你是谁？",
        "你能做什么？",
        "帮我写一段Python代码",
    ]
    
    for q in questions:
        print(f"用户: {q}")
        response, history = chat(q, history, system_prompt=system)
        print(f"助手: {response}")
        print("-" * 40)
