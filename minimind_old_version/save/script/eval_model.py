import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from MiniMind import MiniMindForCausalLM
from tqdm import tqdm

# ============ 加载模型 ============
model_path = "./minimind_hf"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = MiniMindForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
model = model.to(torch.bfloat16)
model.eval()


# ============ 构建 prompt ============
def build_prompt(question, choices, subject):
    """构建 MMLU 标准格式的 prompt"""
    choice_labels = ["A", "B", "C", "D"]
    
    prompt = f"The following is a multiple choice question about {subject}.\n\n"
    prompt += f"Question: {question}\n"
    for label, choice in zip(choice_labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    
    return prompt


# ============ 预测答案 ============
def predict(question, choices, subject):
    """
    通过比较 A/B/C/D 四个选项的生成概率来选择答案
    """
    prompt = build_prompt(question, choices, subject)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    choice_labels = ["A", "B", "C", "D"]
    
    # 获取每个选项的概率
    with torch.no_grad():
        outputs = model(**inputs)
        logits  = outputs.logits   # [1, seq_len, vocab_size]
    
    # 取最后一个 token 的 logits（Answer: 后面那个位置）
    last_logits = logits[0, -1, :]   # [vocab_size]
    
    # 找 A B C D 对应的 token id
    choice_ids = [
        tokenizer.encode(label, add_special_tokens=False)[0]
        for label in choice_labels
    ]
    
    # 取 A B C D 四个位置的 logits
    choice_logits = last_logits[choice_ids]   # [4]
    
    # 概率最大的就是预测答案
    pred_idx = choice_logits.argmax().item()
    
    return choice_labels[pred_idx]


# ============ 评测主函数 ============
def evaluate_mmlu(subjects=None, num_few_shot=0):
    """
    subjects: 要测试的学科列表，None表示测全部
    num_few_shot: few-shot 示例数量（0表示zero-shot）
    """
    
    # MMLU 57个学科
    all_subjects = [
        "abstract_algebra", "anatomy", "astronomy",
        "business_ethics", "clinical_knowledge",
        "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics",
        "college_medicine", "college_physics",
        "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering",
        "elementary_mathematics", "formal_logic",
        "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology",
        "high_school_statistics", "high_school_us_history",
        "high_school_world_history", "human_aging",
        "human_sexuality", "international_law",
        "jurisprudence", "logical_fallacies",
        "machine_learning", "management", "marketing",
        "medical_genetics", "miscellaneous",
        "moral_disputes", "moral_scenarios", "nutrition",
        "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine",
        "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy",
        "virology", "world_religions",
    ]
    
    if subjects is None:
        subjects = all_subjects
    
    total_correct = 0
    total_count   = 0
    subject_results = {}
    
    for subject in subjects:
        print(f"\n评测学科: {subject}")
        
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test")
        except:
            print(f"  跳过 {subject}（加载失败）")
            continue
        
        correct = 0
        count   = 0
        
        answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        
        for item in tqdm(dataset, desc=subject):
            question = item["question"]
            choices  = item["choices"]
            label    = answer_map[item["answer"]]   # 正确答案
            
            pred = predict(question, choices, subject)
            
            if pred == label:
                correct += 1
            count += 1
        
        acc = correct / count if count > 0 else 0
        subject_results[subject] = acc
        
        total_correct += correct
        total_count   += count
        
        print(f"  {subject}: {correct}/{count} = {acc:.3f}")
    
    # 总分
    total_acc = total_correct / total_count if total_count > 0 else 0
    
    print("\n" + "="*50)
    print("MMLU 评测结果：")
    print("="*50)
    
    for subject, acc in sorted(subject_results.items(), key=lambda x: x[1]):
        print(f"  {subject:45s}: {acc:.3f}")
    
    print("="*50)
    print(f"  总准确率: {total_acc:.3f}  ({total_correct}/{total_count})")
    print("="*50)
    
    return total_acc, subject_results


# ============ 运行评测 ============
if __name__ == "__main__":
    # 快速测试：只测几个学科
    quick_subjects = [
        "high_school_mathematics",
        "high_school_physics",
        "computer_security",
        "machine_learning",
    ]
    
    # 测试部分学科
    acc, results = evaluate_mmlu(subjects=quick_subjects)
    
    # 测试全部57个学科（时间较长）
    # acc, results = evaluate_mmlu()
    
    print(f"\n最终 MMLU 分数: {acc:.4f}")
