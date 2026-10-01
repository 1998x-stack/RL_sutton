# 07_5.8_贴现意识的重要性采样

"""
Lecture: /05._蒙特卡罗方法
Content: 07_5.8_贴现意识的重要性采样
"""

import numpy as np
from typing import Dict, Tuple, List

class DiscountingAwareIS:
    """
    贴现意识的重要性采样类，包含用于求解马尔可夫决策过程（MDP）的策略评估和改进算法。

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
                 behavior_policy: Dict[int, Dict[int, float]]):
        """
        初始化贴现意识的重要性采样类。

        参数:
            states: 状态集合
            actions: 动作集合
            transition_probabilities: 状态转移概率矩阵
            rewards: 奖励矩阵
            gamma: 折扣因子
            behavior_policy: 行为策略
        """
        self.states = states
        self.actions = actions
        self.transition_probabilities = transition_probabilities
        self.rewards = rewards
        self.gamma = gamma
        self.behavior_policy = behavior_policy
        self.q_value = {(s, a): 0 for s in states for a in actions}
        self.cumulative_weights = {(s, a): 0 for s in states for a in actions}
        self.target_policy = {s: np.random.choice(actions) for s in states}

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
        for t, (state, action, reward) in enumerate(reversed(episode)):
            G = reward + self.gamma * G
            self.cumulative_weights[(state, action)] += W
            self.q_value[(state, action)] += (W / self.cumulative_weights[(state, action)]) * (G - self.q_value[(state, action)])
            self.target_policy[state] = max(self.actions, key=lambda a: self.q_value[(state, a)])
            if action != self.target_policy[state]:
                break
            W *= 1.0 / self.behavior_policy[state][action]

    def discounting_aware_is(self, num_episodes: int) -> None:
        """
        贴现意识的重要性采样算法，通过策略评估和策略改进找到最优策略。

        参数:
            num_episodes: 运行的序列数量
        """
        for _ in range(num_episodes):
            episode = self.generate_episode()
            self.update_q_value(episode)
