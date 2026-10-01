# 04_3.5_策略和值函数

"""
Lecture: /03._有限马尔可夫决策过程
Content: 04_3.5_策略和值函数
"""

import numpy as np
from typing import List, Dict

class BellmanOptimalEquations:
    """
    用于计算贝尔曼最优方程的类

    Attributes:
        states: 状态空间的列表
        actions: 动作空间的列表
        transition_probabilities: 状态转移概率字典
        rewards: 奖励函数字典
        gamma: 折扣因子
        state_values: 状态值函数
        action_values: 行动值函数
    """

    def __init__(self, states: List[str], actions: List[str], transition_probabilities: Dict[str, Dict[str, Dict[str, float]]], rewards: Dict[str, Dict[str, Dict[str, float]]], gamma: float = 0.9) -> None:
        """
        初始化贝尔曼最优方程类
        
        Args:
            states: 状态空间的列表
            actions: 动作空间的列表
            transition_probabilities: 状态转移概率字典，格式为{状态: {动作: {下一个状态: 概率}}}
            rewards: 奖励函数字典，格式为{状态: {动作: {下一个状态: 奖励}}}
            gamma: 折扣因子，默认为0.9
        """
        self.states = states
        self.actions = actions
        self.transition_probabilities = transition_probabilities
        self.rewards = rewards
        self.gamma = gamma
        self.state_values = {state: 0.0 for state in states}
        self.action_values = {state: {action: 0.0 for action in actions} for state in states}

    def bellman_optimality_update(self) -> None:
        """
        使用贝尔曼最优方程更新状态值函数
        """
        new_state_values = {}
        for state in self.states:
            max_value = float('-inf')
            for action in self.actions:
                q_value = sum(self.transition_probabilities[state][action][next_state] * 
                              (self.rewards[state][action][next_state] + self.gamma * self.state_values[next_state])
                              for next_state in self.states)
                if q_value > max_value:
                    max_value = q_value
            new_state_values[state] = max_value
        self.state_values = new_state_values

    def compute_optimal_policy(self) -> Dict[str, str]:
        """
        计算最优策略
        
        Returns:
            最优策略字典，格式为{状态: 动作}
        """
        policy = {}
        for state in self.states:
            max_value = float('-inf')
            best_action = None
            for action in self.actions:
                q_value = sum(self.transition_probabilities[state][action][next_state] * 
                              (self.rewards[state][action][next_state] + self.gamma * self.state_values[next_state])
                              for next_state in self.states)
                if q_value > max_value:
                    max_value = q_value
                    best_action = action
            policy[state] = best_action
        return policy

    def value_iteration(self, epsilon: float = 1e-6) -> None:
        """
        执行值迭代算法，直到收敛
        
        Args:
            epsilon: 收敛阈值，默认为1e-6
        """
        while True:
            old_state_values = self.state_values.copy()
            self.bellman_optimality_update()
            delta = max(abs(old_state_values[state] - self.state_values[state]) for state in self.states)
            if delta < epsilon:
                break

def main():
    """
    主函数，测试贝尔曼最优方程类
    """
    states = ['s1', 's2', 's3']
    actions = ['a1', 'a2']
    transition_probabilities = {
        's1': {
            'a1': {'s1': 0.1, 's2': 0.9, 's3': 0.0},
            'a2': {'s1': 0.0, 's2': 0.0, 's3': 1.0}
        },
        's2': {
            'a1': {'s1': 0.1, 's2': 0.8, 's3': 0.1},
            'a2': {'s1': 0.0, 's2': 0.9, 's3': 0.1}
        },
        's3': {
            'a1': {'s1': 0.0, 's2': 0.0, 's3': 1.0},
            'a2': {'s1': 0.0, 's2': 0.0, 's3': 1.0}
        }
    }
    rewards = {
        's1': {
            'a1': {'s1': 0, 's2': 10, 's3': 0},
            'a2': {'s1': 0, 's2': 0, 's3': 50}
        },
        's2': {
            'a1': {'s1': 0, 's2': -10, 's3': 10},
            'a2': {'s1': 0, 's2': -10, 's3': 10}
        },
        's3': {
            'a1': {'s1': 0, 's2': 0, 's3': 0},
            'a2': {'s1': 0, 's2': 0, 's3': 0}
        }
    }

    bellman = BellmanOptimalEquations(states, actions, transition_probabilities, rewards)
    bellman.value_iteration()
    
    print("最优状态值函数:")
    for state, value in bellman.state_values.items():
        print(f"状态 {state}: {value}")

    optimal_policy = bellman.compute_optimal_policy()
    print("\n最优策略:")
    for state, action in optimal_policy.items():
        print(f"状态 {state}: 动作 {action}")

if __name__ == "__main__":
    main()
