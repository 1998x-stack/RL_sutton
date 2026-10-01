# 07_2.8_梯度强盗算法

"""
Lecture: /02._表格解法方法
Content: 07_2.8_梯度强盗算法
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

class GradientBandit:
    """梯度强盗算法的实现类

    Attributes:
        num_arms: 动作数量，即拉杆数量
        preferences: 各拉杆的偏好值
        action_probs: 各拉杆的选择概率
        avg_reward: 平均奖励
        alpha: 步长参数
    """

    def __init__(self, num_arms: int = 10, alpha: float = 0.1) -> None:
        """
        初始化梯度强盗算法实例
        
        Args:
            num_arms: 动作数量，默认为10
            alpha: 步长参数，默认为0.1
        """
        self.num_arms = num_arms
        self.alpha = alpha
        self.preferences = np.zeros(num_arms)  # 各拉杆的偏好值
        self.action_probs = np.ones(num_arms) / num_arms  # 各拉杆的选择概率
        self.avg_reward = 0.0  # 平均奖励
        self.true_values = np.random.randn(num_arms)  # 各拉杆的真实奖励期望值

    def select_action(self) -> int:
        """
        根据当前的选择概率选择动作
        
        Returns:
            选择的动作索引
        """
        return np.random.choice(self.num_arms, p=self.action_probs)

    def update_preferences(self, action: int, reward: float) -> None:
        """
        更新动作的偏好值和选择概率
        
        Args:
            action: 动作索引
            reward: 动作获得的奖励
        """
        self.avg_reward += (reward - self.avg_reward) / (np.sum(self.action_probs) + 1)
        self.preferences[action] += self.alpha * (reward - self.avg_reward) * (1 - self.action_probs[action])
        for a in range(self.num_arms):
            if a != action:
                self.preferences[a] -= self.alpha * (reward - self.avg_reward) * self.action_probs[a]
        self.action_probs = np.exp(self.preferences) / np.sum(np.exp(self.preferences))

    def run(self, steps: int) -> List[Tuple[int, float]]:
        """
        运行梯度强盗算法
        
        Args:
            steps: 运行的步数
        
        Returns:
            每一步的动作和奖励
        """
        results = []
        for _ in range(steps):
            action = self.select_action()
            reward = np.random.randn() + self.true_values[action]
            self.update_preferences(action, reward)
            results.append((action, reward))
        return results

def main():
    """
    主函数，执行梯度强盗算法并打印结果
    """
    num_arms = 10
    steps = 1000
    alpha = 0.1

    bandit = GradientBandit(num_arms, alpha)
    results = bandit.run(steps)

    # 转换为DataFrame并打印结果
    df = pd.DataFrame(results, columns=['Action', 'Reward'])
    print(df.describe())

    print(f"真实值: {bandit.true_values}")
    print(f"选择概率: {bandit.action_probs}")
    print(f"偏好值: {bandit.preferences}")

    return df

# Run the main function and save results
df_results = main()
df_results.to_csv("dataset/gradient_bandit_results.csv", index=False)
