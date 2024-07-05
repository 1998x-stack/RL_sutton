# 00_2.1_多臂强盗问题

"""
Lecture: /02._表格解法方法
Content: 00_2.1_多臂强盗问题
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List

class MultiArmedBandit:
    """多臂强盗问题的实现类

    Attributes:
        num_arms: 拉杆数量
        true_values: 各拉杆的真实奖励期望值
        estimated_values: 各拉杆的估计奖励期望值
        action_counts: 各拉杆的被选择次数
        epsilon: 探索概率
    """

    def __init__(self, num_arms: int, epsilon: float = 0.1) -> None:
        """
        初始化多臂强盗问题实例
        
        Args:
            num_arms: 拉杆数量
            epsilon: 探索概率，默认为0.1
        """
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.true_values = np.random.randn(num_arms)  # 各拉杆的真实奖励期望值
        self.estimated_values = np.zeros(num_arms)  # 各拉杆的估计奖励期望值
        self.action_counts = np.zeros(num_arms)  # 各拉杆的被选择次数

    def select_action(self) -> int:
        """
        根据ε-贪婪策略选择动作
        
        Returns:
            选择的动作索引
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_arms)  # 随机选择动作
        else:
            return np.argmax(self.estimated_values)  # 选择估计值最大的动作

    def update_estimates(self, action: int, reward: float) -> None:
        """
        更新动作的估计值
        
        Args:
            action: 动作索引
            reward: 动作获得的奖励
        """
        self.action_counts[action] += 1
        self.estimated_values[action] += (reward - self.estimated_values[action]) / self.action_counts[action]

    def run(self, steps: int) -> List[float]:
        """
        运行多臂强盗算法
        
        Args:
            steps: 运行的步数
        
        Returns:
            每一步的奖励
        """
        rewards = []
        for _ in range(steps):
            action = self.select_action()
            reward = np.random.randn() + self.true_values[action]
            self.update_estimates(action, reward)
            rewards.append(reward)
        return rewards

def main():
    """
    主函数，执行多臂强盗算法并打印结果
    """
    num_arms = 10
    steps = 1000
    epsilon = 0.1

    bandit = MultiArmedBandit(num_arms, epsilon)
    rewards = bandit.run(steps)

    print(f"真实值: {bandit.true_values}")
    print(f"估计值: {bandit.estimated_values}")
    print(f"选择次数: {bandit.action_counts}")
    print(f"总奖励: {np.sum(rewards)}")
    print(f"平均奖励: {np.mean(rewards)}")

    # 可视化结果
    plt.figure(figsize=(12, 8))
    plt.plot(rewards)
    plt.xlabel('步骤')
    plt.ylabel('奖励')
    plt.title('多臂强盗问题奖励变化')
    plt.show()

if __name__ == "__main__":
    main()