# 06_3.7_最优性和近似

"""
Lecture: /03._有限马尔可夫决策过程
Content: 06_3.7_最优性和近似
"""

import numpy as np
from typing import List, Dict

class LinearApproximation:
    """
    线性近似值函数类

    该类使用线性函数逼近状态值函数。
    
    Attributes:
        features: 状态的特征表示
        weights: 线性近似的权重参数
        alpha: 学习率
        gamma: 折扣因子
    """

    def __init__(self, num_features: int, alpha: float = 0.01, gamma: float = 0.9) -> None:
        """
        初始化线性近似类
        
        Args:
            num_features: 状态特征的数量
            alpha: 学习率，默认为0.01
            gamma: 折扣因子，默认为0.9
        """
        self.num_features = num_features
        self.weights = np.zeros(num_features)
        self.alpha = alpha
        self.gamma = gamma

    def value(self, features: np.ndarray) -> float:
        """
        计算状态的值函数
        
        Args:
            features: 状态的特征表示
        
        Returns:
            该状态的值
        """
        return np.dot(self.weights, features)

    def update(self, features: np.ndarray, reward: float, next_features: np.ndarray, done: bool) -> None:
        """
        更新权重参数
        
        Args:
            features: 当前状态的特征表示
            reward: 当前动作获得的奖励
            next_features: 下一个状态的特征表示
            done: 当前情节是否结束
        """
        current_value = self.value(features)
        next_value = self.value(next_features) if not done else 0.0
        target = reward + self.gamma * next_value
        td_error = target - current_value
        self.weights += self.alpha * td_error * features

    def reset(self) -> None:
        """重置权重参数"""
        self.weights = np.zeros(self.num_features)

def main():
    """
    主函数，测试线性近似值函数类
    """
    num_features = 5
    linear_approx = LinearApproximation(num_features)

    # 测试特征向量
    state_features = np.random.rand(num_features)
    next_state_features = np.random.rand(num_features)

    # 打印初始值函数
    print(f"初始状态值: {linear_approx.value(state_features)}")

    # 模拟更新
    reward = 1.0
    done = False
    linear_approx.update(state_features, reward, next_state_features, done)

    # 打印更新后的值函数
    print(f"更新后的状态值: {linear_approx.value(state_features)}")

    # 重置权重参数
    linear_approx.reset()
    print(f"重置后的状态值: {linear_approx.value(state_features)}")

if __name__ == "__main__":
    main()