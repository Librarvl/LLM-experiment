import numpy as np
import random

class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions # 动作列表: ['up', 'down', 'left', 'right']
        self.lr = learning_rate # alpha: 学习率
        self.gamma = reward_decay # gamma: 折扣因⼦
        self.epsilon = e_greedy # epsilon: 贪婪度 (90% ⼏率选最优，10% 乱⾛)
    
        # 初始化 Q 表 (使⽤字典，Key为坐标，Value为各动作的分数)
        # 例如: '0,0': [0.0, 0.0, 0.0, 0.0]
        self.q_table = {}

    def check_state_exist(self, state):
        """如果状态第⼀次出现，在 Q 表中初始化它"""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(self.actions)
    
    def choose_action(self, state):
        """根据 Epsilon-Greedy 策略选择动作"""
        self.check_state_exist(state)
        # 探索 (Exploration): 随机乱⾛
        if np.random.uniform() > self.epsilon:
            action = np.random.choice(self.actions)
        # 利⽤ (Exploitation): 选分最⾼的
        else:
            state_action = self.q_table[state]
            # 可能会有多个最⼤值，随机选⼀个防⽌死循环
            max_val = max(state_action)
            best_actions = [i for i, v in enumerate(state_action) if v == max_val]
            action_idx = np.random.choice(best_actions)
            action = self.actions[action_idx]
        return action
    

    def learn(self, s, a_idx, r, s_next):
        """
        核⼼学习步骤 (对应上⾯的公式)
        s: 当前状态
        a_idx: 当前动作的索引
        r: 获得的奖励
        s_next: 下⼀个状态
        """
        self.check_state_exist(s_next)
        # 1. 预测值 Q(S, A)
        q_predict = self.q_table[s][a_idx]

        # 2. ⽬标值 Q_target = R + gamma * max(Q(S', a'))
        if s_next != 'terminal': # 如果不是终点
            q_target = r + self.gamma * max(self.q_table[s_next])
        else:
            q_target = r # 如果是终点，没有未来
        # 3. 更新 Q 表
        # Q(S, A) += alpha * (Q_target - Q_predict)
        self.q_table[s][a_idx] += self.lr * (q_target - q_predict)


# --- 模拟训练循环 ---
if __name__ == "__main__":
    agent = QLearningAgent(actions=['up', 'down', 'left', 'right'])
    # 假设训练 100 回合
    for episode in range(100):
        state = '0,0' # 起点
        is_terminated = False
        
        while not is_terminated:
            # 1. 选动作
            action = agent.choose_action(state)
            action_idx = agent.actions.index(action)
            # 2. 执⾏动作，获取环境反馈 (这⾥简化模拟)
            # next_state, reward, done = env.step(action)
            # 假设我们从 (0,0) 往右⾛到了 (1,0)，奖励 -1
            next_state = '1,0'
            reward = -1
            if next_state == 'goal':
                reward = 100
            is_terminated = True

            # 3. 学习 (更新 Q 表)
            agent.learn(state, action_idx, reward, next_state)
            
            # 4. 移动到新状态
            state = next_state

    print("训练后的 Q 表:")
    print(agent.q_table)
