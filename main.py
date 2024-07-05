structure = {
    "01._引言": [
        "1.1_强化学习",
        "1.2_示例",
        "1.3_强化学习的要素",
        "1.4_限制与范围",
        "1.5_扩展示例：井字游戏",
        "1.6_小结",
        "1.7_强化学习的早期历史"
    ],
    "02._表格解法方法": [
        "2.1_多臂强盗问题",
        "2.2_动作值方法",
        "2.3_10臂测试平台",
        "2.4_增量实现",
        "2.5_跟踪非平稳问题",
        "2.6_乐观初始值",
        "2.7_上置信度界动作选择",
        "2.8_梯度强盗算法",
        "2.9_关联搜索（上下文强盗）",
        "2.10_小结"
    ],
    "03._有限马尔可夫决策过程": [
        "3.1_代理-环境接口",
        "3.2_目标和奖励",
        "3.3_回报和情节",
        "3.4_情节和持续任务的统一表示",
        "3.5_策略和值函数",
        "3.6_最优策略和最优值函数",
        "3.7_最优性和近似",
        "3.8_小结"
    ],
    "04._动态规划": [
        "4.1_策略评估（预测）",
        "4.2_策略改进",
        "4.3_策略迭代",
        "4.4_值迭代",
        "4.5_异步动态规划",
        "4.6_广义策略迭代",
        "4.7_动态规划的效率",
        "4.8_小结"
    ],
    "05._蒙特卡罗方法": [
        "5.1_蒙特卡罗预测",
        "5.2_动作值的蒙特卡罗估计",
        "5.3_蒙特卡罗控制",
        "5.4_无探索启动的蒙特卡罗控制",
        "5.5_离策略预测通过重要性采样",
        "5.6_增量实现",
        "5.7_离策略蒙特卡罗控制",
        "5.8_贴现意识的重要性采样",
        "5.9_每决策重要性采样",
        "5.10_小结"
    ],
    "06._时序差分学习": [
        "6.1_TD预测",
        "6.2_TD预测方法的优势",
        "6.3_TD(0)的最优性",
        "6.4_Sarsa：在策略TD控制",
        "6.5_Q学习：离策略TD控制",
        "6.6_预期Sarsa",
        "6.7_最大化偏差和双重学习",
        "6.8_游戏、后状态和其他特殊情况",
        "6.9_小结"
    ],
    "07._n步引导": [
        "7.1_n步TD预测",
        "7.2_n步Sarsa",
        "7.3_n步离策略学习",
        "7.4_使用控制变量的每决策方法",
        "7.5_无重要性采样的离策略学习：n步树备份算法",
        "7.6_统一算法：n步Q(λ)",
        "7.7_小结"
    ],
    "08._使用表格方法进行规划和学习": [
        "8.1_模型和规划",
        "8.2_Dyna：集成规划、行动和学习",
        "8.3_当模型错误时",
        "8.4_优先扫描",
        "8.5_预期与样本更新",
        "8.6_轨迹采样",
        "8.7_实时动态规划",
        "8.8_决策时的规划",
        "8.9_启发式搜索",
        "8.10_展望算法",
        "8.11_蒙特卡罗树搜索",
        "8.12_章节小结",
        "8.13_第一部分小结"
    ],
    "09._近似解法方法": [
        "9.1_使用近似的在策略预测",
        "9.2_预测目标（VE）",
        "9.3_随机梯度和半梯度方法",
        "9.4_线性方法",
        {
            "9.5_线性方法的特征构造": [
                "9.5.1_多项式",
                "9.5.2_傅里叶基",
                "9.5.3_粗编码",
                "9.5.4_平铺编码",
                "9.5.5_径向基函数"
            ]
        },
        "9.6_手动选择步长参数",
        "9.7_非线性函数近似：人工神经网络",
        "9.8_最小二乘TD",
        "9.9_基于内存的函数近似",
        "9.10_基于核的函数近似",
        "9.11_深入了解在策略学习：兴趣和强调",
        "9.12_小结"
    ],
    "10._使用近似的在策略控制": [
        "10.1_情节半梯度控制",
        "10.2_半梯度n步Sarsa",
        "10.3_持续任务的新问题设置：平均奖励",
        "10.4_抛弃贴现设置",
        "10.5_差分半梯度n步Sarsa",
        "10.6_小结"
    ],
    "11._使用近似的离策略方法": [
        "11.1_半梯度方法",
        "11.2_离策略发散的示例",
        "11.3_致命三合一",
        "11.4_线性值函数几何",
        "11.5_贝尔曼误差中的梯度下降",
        "11.6_贝尔曼误差不可学习",
        "11.7_梯度-TD方法",
        "11.8_强调-TD方法",
        "11.9_降低方差",
        "11.10_小结"
    ],
    "12._资格迹": [
        "12.1_λ回报",
        "12.2_TD(λ)",
        "12.3_n步截断λ回报方法",
        "12.4_重新更新：在线λ回报算法",
        "12.5_真正的在线TD(λ)",
        "12.6_荷兰迹在蒙特卡罗学习中的应用",
        "12.7_Sarsa(λ)",
        "12.8_变量λ和β",
        "12.9_控制变量的离策略迹",
        "12.10_从Watkins’s Q(λ)到树备份(λ)",
        "12.11_稳定的离策略迹方法",
        "12.12_实现问题",
        "12.13_结论"
    ],
    "13._策略梯度方法": [
        "13.1_策略近似及其优势",
        "13.2_策略梯度定理",
        "13.3_REINFORCE：蒙特卡罗策略梯度",
        "13.4_带基线的REINFORCE",
        "13.5_行动者-评论者方法",
        "13.6_持续问题的策略梯度",
        "13.7_连续动作的策略参数化",
        "13.8_小结"
    ],
    "14._心理学": [
        "14.1_预测与控制",
        "14.2_经典条件反射",
        {
            "14.2.1_阻碍和高阶条件反射": [
                "14.2.2_Rescorla-Wagner模型",
                "14.2.3_TD模型",
                "14.2.4_TD模型模拟"
            ]
        },
        "14.3_工具性条件反射",
        "14.4_延迟强化",
        "14.5_认知地图",
        "14.6_习惯性和目标导向行为",
        "14.7_小结"
    ],
    "15._神经科学": [
        "15.1_神经科学基础",
        "15.2_奖励信号、强化信号、值和预测误差",
        "15.3_奖励预测误差假说",
        "15.4_多巴胺",
        "15.5_奖励预测误差假说的实验支持",
        "15.6_TD误差/多巴胺对应",
        "15.7_神经行动者-评论者",
        "15.8_行动者和评论者学习规则",
        "15.9_享乐神经元",
        "15.10_集体强化学习",
        "15.11_大脑中的基于模型的方法",
        "15.12_成瘾",
        "15.13_小结"
    ],
    "16._应用与案例研究": [
        "16.1_TD-Gammon",
        "16.2_Samuel的跳棋玩家",
        "16.3_Watson的每日双重赌注",
        "16.4_优化记忆控制",
        "16.5_人类水平的视频游戏玩法",
        {
            "16.6_围棋游戏的掌握": [
                "16.6.1_AlphaGo",
                "16.6.2_AlphaGo_Zero"
            ]
        },
        "16.7_个性化网络服务",
        "16.8_热力上升"
    ],
    "17._前沿": [
        "17.1_通用值函数和辅助任务",
        "17.2_通过选项进行时间抽象",
        "17.3_观察与状态",
        "17.4_设计奖励信号",
        "17.5_遗留问题",
        "17.6_人工智能的未来"
    ],
    "附录": [
        "参考文献",
        "索引"
    ]
}




import os
from typing import Dict, Any

def create_directories_and_files(
        base_path: str, 
        structure: Dict[str, Any], 
        readme_file, 
        parent_path: str = "", 
        level: int = 1
    ):
    heading = "#" * level

    for key, value in structure.items():
        current_path = os.path.join(base_path, key.replace(" ", "_").replace("/", "_").replace("-", "_"))

        # 创建目录
        os.makedirs(current_path, exist_ok=True)

        # 在README中添加章节标题
        if parent_path:
            readme_file.write(f"{heading} {parent_path}/{key}\n\n")
        else:
            readme_file.write(f"{heading} {key}\n\n")

        # 递归调用创建子目录和文件
        if isinstance(value, dict) and value:
            create_directories_and_files(
                current_path, 
                value, 
                readme_file, 
                parent_path + "/" + key if parent_path else key, 
                level + 1
            )
        elif isinstance(value, list) and value:
            for idx, item in enumerate(value):
                if isinstance(item, dict) and item:
                    create_directories_and_files(
                        current_path, 
                        item, 
                        readme_file, 
                        parent_path + "/" + key if parent_path else key, 
                        level + 1
                    )
                else:
                    item = f"{idx:02d}_{item}"
                    file_name = item.replace(" ", "_").replace("/", "_").replace("-", "_") + ".py"
                    file_path = os.path.join(current_path, file_name)
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(f"# {item}\n\n")
                        file.write(f'"""\nLecture: {parent_path}/{key}\nContent: {item}\n"""\n\n')

                    # 在README中添加文件链接
                    item_clean = item.replace(" ", "_").replace("/", "_").replace("-", "_")
                    parent_clean = parent_path.replace(" ", "_").replace("/", "_").replace("-", "_")
                    key_clean = key.replace(" ", "_").replace("/", "_").replace("-", "_")
                    readme_file.write(f"- [{item}](./{parent_clean}/{key_clean}/{item_clean}.py)\n")
                    
                    
                    file_name = item.replace(" ", "_").replace("/", "_").replace("-", "_") + ".md"
                    file_path = os.path.join(current_path, file_name)
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(f"# {item}\n\n")
                        file.write(f'"""\nLecture: {parent_path}/{key}\nContent: {item}\n"""\n\n')

                    # 在README中添加文件链接
                    item_clean = item.replace(" ", "_").replace("/", "_").replace("-", "_")
                    parent_clean = parent_path.replace(" ", "_").replace("/", "_").replace("-", "_")
                    key_clean = key.replace(" ", "_").replace("/", "_").replace("-", "_")
                    readme_file.write(f"- [{item}](./{parent_clean}/{key_clean}/{item_clean}.md)\n")
                    
                    
                    
                    file_name = item.replace(" ", "_").replace("/", "_").replace("-", "_") + ".cpp"
                    file_path = os.path.join(current_path, file_name)
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(f"// {item}\n\n")
                        file.write(f'/*\n\nLecture: {parent_path}/{key}\nContent: {item}\n\n*/\n')

                    # 在README中添加文件链接
                    item_clean = item.replace(" ", "_").replace("/", "_").replace("-", "_")
                    parent_clean = parent_path.replace(" ", "_").replace("/", "_").replace("-", "_")
                    key_clean = key.replace(" ", "_").replace("/", "_").replace("-", "_")
                    readme_file.write(f"- [{item}](./{parent_clean}/{key_clean}/{item_clean}.cpp)\n")
        else:
            # 创建文件并写入初始内容
            file_name = key.replace(" ", "_").replace("/", "_").replace("-", "_") + ".py"
            file_path = os.path.join(current_path, file_name)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(f"# {key}\n\n")
                file.write(f'"""\nLecture: {parent_path}/{key}\nContent: {key}\n"""\n\n')

            # 在README中添加文件链接
            parent_clean = parent_path.replace(" ", "_").replace("/", "_").replace("-", "_")
            key_clean = key.replace(" ", "_").replace("/", "_").replace("-", "_")
            readme_file.write(f"- [{key}](./{parent_clean}/{key_clean}/{file_name})\n")
            
            
            file_name = key.replace(" ", "_").replace("/", "_").replace("-", "_") + ".md"
            file_path = os.path.join(current_path, file_name)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(f"# {key}\n\n")
                file.write(f'"""\nLecture: {parent_path}/{key}\nContent: {key}\n"""\n\n')

            # 在README中添加文件链接
            parent_clean = parent_path.replace(" ", "_").replace("/", "_").replace("-", "_")
            key_clean = key.replace(" ", "_").replace("/", "_").replace("-", "_")
            readme_file.write(f"- [{key}](./{parent_clean}/{key_clean}/{file_name})\n")
            
            
            file_name = key.replace(" ", "_").replace("/", "_").replace("-", "_") + ".cpp"
            file_path = os.path.join(current_path, file_name)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(f"// {key}\n\n")
                file.write(f'/*\n\nLecture: {parent_path}/{key}\nContent: {key}\n\n*/\n')

            # 在README中添加文件链接
            parent_clean = parent_path.replace(" ", "_").replace("/", "_").replace("-", "_")
            key_clean = key.replace(" ", "_").replace("/", "_").replace("-", "_")
            readme_file.write(f"- [{key}](./{parent_clean}/{key_clean}/{file_name})\n")

        # 添加空行以分隔不同的章节
        readme_file.write("\n")

def main():
    root_dir = './'
    # 创建根目录
    os.makedirs(root_dir, exist_ok=True)

    # 创建 README.md 文件
    with open(os.path.join(root_dir, "README.md"), 'w', encoding='utf-8') as readme_file:
        readme_file.write("# SUTTON_RL\n\n")
        readme_file.write("这是一个关于SUTTON_RL的目录结构。\n\n")
        create_directories_and_files(root_dir, structure, readme_file)

    print("目录和文件结构已生成，并创建 README.md 文件。")

if __name__ == "__main__":
    main()