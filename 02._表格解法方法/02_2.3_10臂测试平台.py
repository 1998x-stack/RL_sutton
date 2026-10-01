# 02_2.3_10臂测试平台

"""
Lecture: /02._表格解法方法
Content: 02_2.3_10臂测试平台
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

class TenArmedTestbed:
    """10臂测试平台的实现类

    Attributes:
        num_arms: 动作数量，即拉杆数量
        true_values: 各拉杆的真实奖励期望值
        estimated_values: 各拉杆的估计奖励期望值
        action_counts: 各拉杆的被选择次数
        epsilon: 探索概率
    """

    def __init__(self, num_arms: int = 10, epsilon: float = 0.1) -> None:
        """
        初始化10臂测试平台实例
        
        Args:
            num_arms: 动作数量，默认为10
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

    def run(self, steps: int) -> List[Tuple[int, float]]:
        """
        运行10臂测试平台
        
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
    主函数，执行10臂测试平台并打印结果
    """
    num_arms = 10
    steps = 10000
    epsilon = 0.1

    testbed = TenArmedTestbed(num_arms, epsilon)
    results = testbed.run(steps)

    # 转换为DataFrame并打印结果
    df = pd.DataFrame(results, columns=['Action', 'Reward'])
    print(df.describe())

    print(f"真实值: {testbed.true_values}")
    print(f"估计值: {testbed.estimated_values}")
    print(f"选择次数: {testbed.action_counts}")

    return df

# Run the main function and save results
df_results = main()
df_results.to_csv("dataset/ten_armed_testbed_results.csv", index=False)
