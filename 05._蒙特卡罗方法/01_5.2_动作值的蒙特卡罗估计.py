# 01_5.2_动作值的蒙特卡罗估计

"""
Lecture: /05._蒙特卡罗方法
Content: 01_5.2_动作值的蒙特卡罗估计
"""

import numpy as np
from typing import Dict, Tuple, List

class MonteCarloActionValue:
    """
    动作值蒙特卡罗估计类，包含用于求解马尔可夫决策过程（MDP）的策略评估和改进算法。

    Attributes:
        states: 状态集合
        actions: 动作集合
        transition_probabilities: 状态转移概率矩阵
        rewards: 奖励矩阵
        gamma: 折扣因子
        epsilon: 探索率
    """

    def __init__(self, states: List[int], actions: List[int],
                 transition_probabilities: Dict[Tuple[int, int, int], float],
                 rewards: Dict[Tuple[int, int, int], float], gamma: float, epsilon: float):
        """
        初始化动作值蒙特卡罗估计类。

        参数:
            states: 状态集合
            actions: 动作集合
            transition_probabilities: 状态转移概率矩阵
            rewards: 奖励矩阵
            gamma: 折扣因子
            epsilon: 探索率
        """
        self.states = states
        self.actions = actions
        self.transition_probabilities = transition_probabilities
        self.rewards = rewards
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_value = {(s, a): 0 for s in states for a in actions}
        self.returns = {(s, a): [] for s in states for a in actions}
        self.policy = {s: np.random.choice(actions) for s in states}

    def generate_episode(self, start_state: int) -> List[Tuple[int, int, float]]:
        """
        生成一个遵循当前策略的完整序列。

        参数:
            start_state: 序列的初始状态

        返回:
            List[Tuple[int, int, float]]: 序列列表，包含状态、动作和奖励
        """
        episode = []
        state = start_state
        while True:
            action = self.policy[state] if np.random.rand() > self.epsilon else np.random.choice(self.actions)
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
        episode.reverse()
        visited = set()
        for state, action, reward in episode:
            G = self.gamma * G + reward
            if (state, action) not in visited:
                self.returns[(state, action)].append(G)
                self.q_value[(state, action)] = np.mean(self.returns[(state, action)])
                visited.add((state, action))

    def policy_improvement(self) -> None:
        """
        根据当前的动作值函数改进策略。
        """
        for state in self.states:
            action_values = {a: self.q_value[(state, a)] for a in self.actions}
            self.policy[state] = max(action_values, key=action_values.get)

    def monte_carlo_control(self, num_episodes: int) -> None:
        """
        蒙特卡罗控制算法，通过策略评估和策略改进找到最优策略。

        参数:
            num_episodes: 运行的序列数量
        """
        for _ in range(num_episodes):
            start_state = np.random.choice(self.states)
            episode = self.generate_episode(start_state)
            self.update_q_value(episode)
            self.policy_improvement()

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
    epsilon = 0.1
    num_episodes = 1000

    mc_av = MonteCarloActionValue(states, actions, transition_probabilities, rewards, gamma, epsilon)
    mc_av.monte_carlo_control(num_episodes)

    print("最终动作值函数:", mc_av.q_value)
    print("最终策略:", mc_av.policy)