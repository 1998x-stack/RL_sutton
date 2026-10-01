# 04_5.5_离策略预测通过重要性采样

"""
Lecture: /05._蒙特卡罗方法
Content: 04_5.5_离策略预测通过重要性采样
"""

import numpy as np
from typing import Dict, Tuple, List

class OffPolicyPredictionIS:
    """
    离策略预测通过重要性采样类，包含用于求解马尔可夫决策过程（MDP）的策略评估算法。

    Attributes:
        states: 状态集合
        actions: 动作集合
        transition_probabilities: 状态转移概率矩阵
        rewards: 奖励矩阵
        gamma: 折扣因子
        behavior_policy: 行为策略
        target_policy: 目标策略
    """

    def __init__(self, states: List[int], actions: List[int],
                 transition_probabilities: Dict[Tuple[int, int, int], float],
                 rewards: Dict[Tuple[int, int, int], float], gamma: float,
                 behavior_policy: Dict[int, Dict[int, float]],
                 target_policy: Dict[int, Dict[int, float]]):
        """
        初始化离策略预测通过重要性采样类。

        参数:
            states: 状态集合
            actions: 动作集合
            transition_probabilities: 状态转移概率矩阵
            rewards: 奖励矩阵
            gamma: 折扣因子
            behavior_policy: 行为策略
            target_policy: 目标策略
        """
        self.states = states
        self.actions = actions
        self.transition_probabilities = transition_probabilities
        self.rewards = rewards
        self.gamma = gamma
        self.behavior_policy = behavior_policy
        self.target_policy = target_policy
        self.q_value = {(s, a): 0 for s in states for a in actions}
        self.cumulative_weights = {(s, a): 0 for s in states for a in actions}

    def generate_episode(self) -> List[Tuple[int, int, float]]:
        """
        生成一个遵循行为策略的完整序列。

        返回:
            List[Tuple[int, int, float]]: 序列列表，包含状态、动作和奖励
        """
        episode = []
        state = np.random.choice(self.states)
        while True:
            action = np.random.choice(self.actions, p=[self.behavior_policy[state][a] for a in self.actions])
            next_state = np.random.choice(self.states, p=[self.transition_probabilities[(state, action, s)] for s in self.states])
            reward = self.rewards[(state, action, next_state)]
            episode.append((state, action, reward))
            if next_state == None:  # 假设终止状态用 None 表示
                break
            state = next_state
        return episode

    def update_q_value(self, episode: List[Tuple[int, int, float]]) -> None:
        """
        使用生成的序列更新动作值函数。

        参数:
            episode: 序列列表，包含状态、动作和奖励
        """
        G = 0
        W = 1
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            self.cumulative_weights[(state, action)] += W
            self.q_value[(state, action)] += (W / self.cumulative_weights[(state, action)]) * (G - self.q_value[(state, action)])
            if action != self.target_policy[state]:
                break
            W *= 1.0 / self.behavior_policy[state][action]

    def off_policy_prediction(self, num_episodes: int) -> None:
        """
        离策略预测算法，通过重要性采样评估目标策略的动作值函数。

        参数:
            num_episodes: 运行的序列数量
        """
        for _ in range(num_episodes):
            episode = self.generate_episode()
            self.update_q_value(episode)

# 测试代码
if __name__ == "__main__":
    # 定义状态、动作、转移概率和奖励
    states = [0, 1, 2, 3]
    actions = [0, 1]
    transition_probabilities = {
        (0, 0, 0): 0.7, (0, 0, 1): 0.3,
        (0, 1, 0): 0.4, (0, 1, 1): 0.6,
        (1, 0, 2): 0.8, (1, 0, 3): 0.2,
        (1, 1, 2): 0.5, (1, 1, 3): 0.5,
        (2, 0, 0): 0.9, (2, 0, 1): 0.1,
        (2, 1, 0): 0.2, (2, 1, 1): 0.8,
        (3, 0, 2): 0.6, (3, 0, 3): 0.4,
        (3, 1, 2): 0.3, (3, 1, 3): 0.7,
    }
    rewards = {
        (0, 0, 0): 5, (0, 0, 1): 10,
        (0, 1, 0): 2, (0, 1, 1): 7,
        (1, 0, 2): 15, (1, 0, 3): 20,
        (1, 1, 2): 9, (1, 1, 3): 13,
        (2, 0, 0): 6, (2, 0, 1): 8,
        (2, 1, 0): 3, (2, 1, 1): 4,
        (3, 0, 2): 12, (3, 0, 3): 18,
        (3, 1, 2): 14, (3, 1, 3): 11,
    }
    gamma = 0.9

    # 定义行为策略和目标策略
    behavior_policy = {
        0: {0: 0.5, 1: 0.5},
        1: {0: 0.5, 1: 0.5},
        2: {0: 0.5, 1: 0.5},
        3: {0: 0.5, 1: 0.5},
    }
    target_policy = {
        0: {0: 1.0, 1: 0.0},
        1: {0: 1.0, 1: 0.0},
        2: {0: 1.0, 1: 0.0},
        3: {0: 1.0, 1: 0.0},
    }

    num_episodes = 1000

    off_policy = OffPolicyPredictionIS(states, actions, transition_probabilities, rewards, gamma, behavior_policy, target_policy)
    off_policy.off_policy_prediction(num_episodes)

    print("最终动作值函数:", off_policy.q_value)
