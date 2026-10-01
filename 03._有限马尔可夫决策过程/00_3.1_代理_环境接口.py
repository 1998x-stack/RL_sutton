# 00_3.1_代理-环境接口

"""
Lecture: /03._有限马尔可夫决策过程
Content: 00_3.1_代理-环境接口
"""

import numpy as np
from typing import Any, Tuple

class AgentEnvironmentInterface:
    """强化学习中的代理-环境接口类

    该类定义了一个代理与环境交互的基本框架，包括状态、动作、奖励和策略的定义。
    
    Attributes:
        state_space: 状态空间
        action_space: 动作空间
        current_state: 当前状态
        policy: 代理的策略
    """

    def __init__(self, state_space: Any, action_space: Any) -> None:
        """
        初始化代理-环境接口
        
        Args:
            state_space: 状态空间
            action_space: 动作空间
        """
        self.state_space = state_space
        self.action_space = action_space
        self.current_state = self.reset()
        self.policy = self.initialize_policy()

    def reset(self) -> Any:
        """
        重置环境，返回初始状态
        
        Returns:
            初始化后的状态
        """
        # 实际应用中，这里应根据具体环境实现重置逻辑
        initial_state = np.random.choice(self.state_space)
        return initial_state

    def initialize_policy(self) -> Any:
        """
        初始化代理的策略
        
        Returns:
            初始化后的策略
        """
        # 实际应用中，这里应根据具体问题定义策略初始化逻辑
        policy = {state: np.random.choice(self.action_space) for state in self.state_space}
        return policy

    def select_action(self, state: Any) -> Any:
        """
        根据当前状态选择动作
        
        Args:
            state: 当前状态
        
        Returns:
            选择的动作
        """
        action = self.policy[state]
        return action

    def step(self, action: Any) -> Tuple[Any, float, bool]:
        """
        执行动作，返回下一个状态、奖励和是否终止
        
        Args:
            action: 代理选择的动作
        
        Returns:
            下一个状态, 奖励, 是否终止 (done)
        """
        # 实际应用中，这里应根据具体环境实现状态转移和奖励计算逻辑
        next_state = np.random.choice(self.state_space)
        reward = np.random.rand()
        done = np.random.choice([True, False])
        return next_state, reward, done

    def update_policy(self, state: Any, action: Any, reward: float, next_state: Any) -> None:
        """
        更新代理的策略
        
        Args:
            state: 当前状态
            action: 执行动作
            reward: 动作获得的奖励
            next_state: 执行动作后的下一个状态
        """
        # 实际应用中，这里应根据具体算法实现策略更新逻辑
        pass

    def run_episode(self, max_steps: int) -> float:
        """
        运行一个回合，返回累积奖励
        
        Args:
            max_steps: 最大步数
        
        Returns:
            累积奖励
        """
        total_reward = 0.0
        state = self.reset()
        for _ in range(max_steps):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            self.update_policy(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            if done:
                break
        return total_reward

def main():
    """
    主函数，测试代理-环境接口类
    """
    state_space = [0, 1, 2, 3, 4]
    action_space = ['left', 'right', 'up', 'down']
    agent_env_interface = AgentEnvironmentInterface(state_space, action_space)

    # 运行一个回合并打印累积奖励
    total_reward = agent_env_interface.run_episode(max_steps=100)
    print(f"累积奖励: {total_reward}")

if __name__ == "__main__":
    main()
