from torch.distributions import Categorical
import torch

# 方式1: 使用概率值 (probs)
probs = torch.tensor([0.1, 0.3, 0.6])
m = Categorical(probs)
print(m)

# 方式2: 使用logits (未归一化的对数概率)
logits = torch.tensor([1.0, 2.0, 3.0])
m = Categorical(logits=logits)
print(m)

# 采样 - 根据概率分布随机选择类别
sample = m.sample()  # 返回 0, 1, 或 2
print(sample)  # tensor(2)  # 概率最大的类别更容易被采样

# 计算对数概率
action = torch.tensor(1)
log_prob = m.log_prob(action)  # 计算选择类别1的对数概率
print(log_prob)

# 获取熵 (衡量不确定性)
entropy = m.entropy()
print(entropy)
