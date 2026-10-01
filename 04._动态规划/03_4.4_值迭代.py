# 03_4.4_值迭代

"""
Lecture: /04._动态规划
Content: 03_4.4_值迭代
"""

import numpy as np
from typing import Dict, Tuple, List

class ValueIteration:
    """
    值迭代类，包含用于求解马尔可夫决策过程（MDP）的值迭代算法。

    Attributes:
        states: 状态集合
        actions: 动作集合
        transition_probabilities: 状态转移概率矩阵
        rewards: 奖励矩阵
        gamma: 折扣因子
        theta: 收敛阈值
    """

    def __init__(self, states: List[int], actions: List[int],
                 transition_probabilities: Dict[Tuple[int, int, int], float],
                 rewards: Dict[Tuple[int, int, int], float], gamma: float, theta: float):
        """
        初始化值迭代类。

        参数:
            states: 状态集合
            actions: 动作集合
            transition_probabilities: 状态转移概率矩阵
            rewards: 奖励矩阵
            gamma: 折扣因子
            theta: 收敛阈值
        """
        self.states = states
        self.actions = actions
        self.transition_probabilities = transition_probabilities
        self.rewards = rewards
        self.gamma = gamma
        self.theta = theta
        self.value_function = {s: 0 for s in states}
        self.policy = {s: actions[0] for s in states}

    def value_iteration(self) -> Tuple[Dict[int, float], Dict[int, int]]:
        """
        值迭代，通过反复更新状态值函数，直到值函数收敛。

        返回:
            Tuple[Dict[int, float], Dict[int, int]]: 最终的状态值函数和策略。
        """
        while True:
            delta = 0
            for s in self.states:
                v = self.value_function[s]
                self.value_function[s] = max(
                    sum(
                        self.transition_probabilities[(s, a, s_)] *
                        (self.rewards[(s, a, s_)] + self.gamma * self.value_function[s_])
                        for s_ in self.states
                    )
                    for a in self.actions
                )
                delta = max(delta, abs(v - self.value_function[s]))
            if delta < self.theta:
                break

        # 提取最优策略
        for s in self.states:
            action_values = {
                a: sum(
                    self.transition_probabilities[(s, a, s_)] *
                    (self.rewards[(s, a, s_)] + self.gamma * self.value_function[s_])
                    for s_ in self.states
                )
                for a in self.actions
            }
            self.policy[s] = max(action_values, key=action_values.get)

        return self.value_function, self.policy

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
    theta = 0.0001

    vi = ValueIteration(states, actions, transition_probabilities, rewards, gamma, theta)
    value_function, policy = vi.value_iteration()

    print("最终状态值函数:", value_function)
    print("最终策略:", policy)
