# 06_2.7_上置信度界动作选择

"""
Lecture: /02._表格解法方法
Content: 06_2.7_上置信度界动作选择
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

class UpperConfidenceBoundBandit:
    """上置信度界(UCB)动作选择方法的实现类

    Attributes:
        num_arms: 动作数量，即拉杆数量
        true_values: 各拉杆的真实奖励期望值
        estimated_values: 各拉杆的估计奖励期望值
        action_counts: 各拉杆的被选择次数
        c: 控制探索程度的常数
    """

    def __init__(self, num_arms: int = 10, c: float = 2.0) -> None:
        """
        初始化上置信度界(UCB)动作选择方法实例
        
        Args:
            num_arms: 动作数量，默认为10
            c: 控制探索程度的常数，默认为2.0
        """
        self.num_arms = num_arms
        self.c = c
        self.true_values = np.random.randn(num_arms)  # 各拉杆的真实奖励期望值
        self.estimated_values = np.zeros(num_arms)  # 各拉杆的估计奖励期望值
        self.action_counts = np.zeros(num_arms)  # 各拉杆的被选择次数
        self.total_steps = 0  # 总时间步数

    def select_action(self) -> int:
        """
        根据UCB策略选择动作
        
        Returns:
            选择的动作索引
        """
        self.total_steps += 1
        ucb_values = self.estimated_values + self.c * np.sqrt(np.log(self.total_steps) / (self.action_counts + 1e-5))
        return np.argmax(ucb_values)

    def update_estimates(self, action: int, reward: float) -> None:
        """
        更新动作的估计值
        
        Args:
            action: 动作索引
            reward: 动作获得的奖励
        """
        self.action_counts[action] += 1
        self.estimated_values[action] += (reward - self.estimated_values[action]) / self.action_counts[action]

    def run(self, steps: int) -> List[Tuple[int, float]]:
        """
        运行上置信度界(UCB)动作选择方法
        
        Args:
            steps: 运行的步数
        
        Returns:
            每一步的动作和奖励
        """
        results = []
        for _ in range(steps):
            action = self.select_action()
            reward = np.random.randn() + self.true_values[action]
            self.update_estimates(action, reward)
            results.append((action, reward))
        return results

def main():
    """
    主函数，执行上置信度界(UCB)动作选择方法并打印结果
    """
    num_arms = 10
    steps = 1000
    c = 2.0

    bandit = UpperConfidenceBoundBandit(num_arms, c)
    results = bandit.run(steps)

    # 转换为DataFrame并打印结果
    df = pd.DataFrame(results, columns=['Action', 'Reward'])
    print(df.describe())

    print(f"真实值: {bandit.true_values}")
    print(f"估计值: {bandit.estimated_values}")
    print(f"选择次数: {bandit.action_counts}")

    return df

# Run the main function and save results
df_results = main()
df_results.to_csv("dataset/ucb_bandit_results.csv", index=False)
