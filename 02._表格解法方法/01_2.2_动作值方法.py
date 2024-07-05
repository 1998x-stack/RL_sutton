# 01_2.2_动作值方法

"""
Lecture: /02._表格解法方法
Content: 01_2.2_动作值方法
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

class ActionValueMethod:
    """动作值方法的实现类

    Attributes:
        num_actions: 动作数量
        epsilon: 探索概率
        true_values: 各动作的真实奖励期望值
        estimated_values: 各动作的估计奖励期望值
        action_counts: 各动作的被选择次数
    """

    def __init__(self, num_actions: int, epsilon: float = 0.1) -> None:
        """
        初始化动作值方法实例
        
        Args:
            num_actions: 动作数量
            epsilon: 探索概率，默认为0.1
        """
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.true_values = np.random.randn(num_actions)  # 各动作的真实奖励期望值
        self.estimated_values = np.zeros(num_actions)  # 各动作的估计奖励期望值
        self.action_counts = np.zeros(num_actions)  # 各动作的被选择次数

    def select_action(self) -> int:
        """
        根据ε-贪婪策略选择动作
        
        Returns:
            选择的动作索引
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)  # 随机选择动作
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
        运行动作值方法
        
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
    主函数，执行动作值方法并打印结果
    """
    num_actions = 10
    steps = 1000
    epsilon = 0.1

    avm = ActionValueMethod(num_actions, epsilon)
    results = avm.run(steps)

    # 转换为DataFrame并打印结果
    df = pd.DataFrame(results, columns=['Action', 'Reward'])
    print(df.describe())

    print(f"真实值: {avm.true_values}")
    print(f"估计值: {avm.estimated_values}")
    print(f"选择次数: {avm.action_counts}")

    # 返回数据表用于显示
    return df

if __name__ == "__main__":
    df_results = main()

