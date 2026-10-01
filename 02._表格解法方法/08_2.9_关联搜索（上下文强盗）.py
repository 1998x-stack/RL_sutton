# 08_2.9_关联搜索（上下文强盗）

"""
Lecture: /02._表格解法方法
Content: 08_2.9_关联搜索（上下文强盗）
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

class ContextualBandit:
    """关联搜索（上下文强盗）算法的实现类

    Attributes:
        num_arms: 动作数量，即广告数量
        context_dim: 上下文向量的维度
        alpha: 控制探索程度的参数
        A: 动作对应的协方差矩阵
        b: 动作对应的奖励向量
    """

    def __init__(self, num_arms: int, context_dim: int, alpha: float = 1.0) -> None:
        """
        初始化关联搜索（上下文强盗）算法实例
        
        Args:
            num_arms: 动作数量
            context_dim: 上下文向量的维度
            alpha: 控制探索程度的参数，默认为1.0
        """
        self.num_arms = num_arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.A = [np.identity(context_dim) for _ in range(num_arms)]  # 协方差矩阵
        self.b = [np.zeros(context_dim) for _ in range(num_arms)]  # 奖励向量

    def select_action(self, context: np.ndarray) -> int:
        """
        根据当前的上下文选择动作
        
        Args:
            context: 当前的上下文向量
        
        Returns:
            选择的动作索引
        """
        p = np.zeros(self.num_arms)
        for a in range(self.num_arms):
            theta = np.linalg.inv(self.A[a]).dot(self.b[a])
            p[a] = context.dot(theta) + self.alpha * np.sqrt(context.dot(np.linalg.inv(self.A[a])).dot(context))
        return np.argmax(p)

    def update(self, action: int, reward: float, context: np.ndarray) -> None:
        """
        更新动作的协方差矩阵和奖励向量
        
        Args:
            action: 动作索引
            reward: 动作获得的奖励
            context: 当前的上下文向量
        """
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context

    def run(self, steps: int, contexts: np.ndarray, rewards: np.ndarray) -> List[Tuple[int, float]]:
        """
        运行关联搜索（上下文强盗）算法
        
        Args:
            steps: 运行的步数
            contexts: 上下文矩阵，每一行是一个上下文向量
            rewards: 奖励矩阵，每一行是一个奖励向量
        
        Returns:
            每一步的动作和奖励
        """
        results = []
        for t in range(steps):
            context = contexts[t]
            action = self.select_action(context)
            reward = rewards[t, action]
            self.update(action, reward, context)
            results.append((action, reward))
        return results

def main():
    """
    主函数，执行关联搜索（上下文强盗）算法并打印结果
    """
    num_arms = 10
    context_dim = 5
    steps = 1000
    alpha = 1.0

    # 生成随机的上下文和奖励数据
    contexts = np.random.randn(steps, context_dim)
    true_rewards = np.random.randn(num_arms, context_dim)
    rewards = contexts.dot(true_rewards.T) + np.random.randn(steps, num_arms) * 0.1

    bandit = ContextualBandit(num_arms, context_dim, alpha)
    results = bandit.run(steps, contexts, rewards)

    # 转换为DataFrame并打印结果
    df = pd.DataFrame(results, columns=['Action', 'Reward'])
    print(df.describe())

    print("最终的协方差矩阵:")
    for a in range(num_arms):
        print(f"动作 {a}: \n{bandit.A[a]}")
    print("最终的奖励向量:")
    for a in range(num_arms):
        print(f"动作 {a}: \n{bandit.b[a]}")

    return df

# Run the main function and save results
df_results = main()
df_results.to_csv("dataset/contextual_bandit_results.csv", index=False)
