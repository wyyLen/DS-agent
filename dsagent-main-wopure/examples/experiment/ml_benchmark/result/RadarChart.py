import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties


def set_chinese_font():
    # 先尝试一些常见的中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']

    for font_name in chinese_fonts:
        font_path = None
        # 尝试查找字体文件
        for font in fm.findSystemFonts():
            if font_name.lower() in os.path.basename(font).lower():
                font_path = font
                break

        if font_path:
            # 添加字体文件
            font_prop = FontProperties(fname=font_path)
            # 设置为全局字体
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
            print(f"成功设置字体: {font_name}")
            return font_prop

    # 如果找不到任何中文字体，创建一个警告
    print("警告：未找到支持中文的字体。图表中的中文可能无法正确显示。")
    return None


# 设置中文字体
font_prop = set_chinese_font()


def ml_bench_radar_chart(models: dict):
    labels = ['IRIS', 'WR', 'BCW', 'Titanic', 'House Prices', 'SCTP', 'ICR', 'SVPC']
    if len(models) == 2:
        colors = ['#1f77b4', '#ff7f0e']
    else:
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, (model, data) in enumerate(models.items()):
        data += data[:1]
        color = colors[i % len(colors)]
        ax.plot(angles, data, linewidth=2, label=model, color=color)
        ax.fill(angles, data, alpha=0.25, color=color)

    ax.set_ylim(0.3, 1.0)
    ax.set_yticklabels([])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)

    plt.title('性能对比', size=16, y=1.1, fontproperties='Microsoft YaHei', fontsize=18)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.15), fontsize=16)

    plt.show()


def cost_visualize(agents: list, costs: list, title: str = "开销对比分析"):
    if len(agents) == 2:
        colors = ['#1f77b4', '#ff7f0e']
    else:
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    plt.bar(agents, costs, color=colors)
    plt.title(f'{title}', fontproperties='Microsoft YaHei', fontsize=16)
    plt.xlabel('实验配置', fontproperties='Microsoft YaHei', fontsize=14)
    plt.ylabel('开销($)', fontproperties='Microsoft YaHei', fontsize=14)
    for i in range(len(costs)):
        plt.text(i, costs[i] + 0.001, str(costs[i]), ha='center')
    plt.show()


def w_plan_radar():
    models = {
        'AutoGen': [1, 0.9375, 0.9175, 0.8785, 0.15, 0.1, 0.7881, 0.1],
        'DataInterpreter': [1, 1, 0.8943, 0.7568, 0.8357, 0.6567, 0.7333, 0.4888],
        'W-Plan': [1, 1, 0.992, 0.939, 0.9095, 0.7833, 0.8418, 0.594]
    }
    ml_bench_radar_chart(models)


def w_lats_radar():
    models = {
        'AutoGen': [1, 0.9375, 0.9175, 0.8785, 0.15, 0.1, 0.7881, 0.1],
        'DataInterpreter': [1, 1, 0.8943, 0.7568, 0.8357, 0.6567, 0.7333, 0.4888],
        'W-LATS': [1, 1, 0.9825, 0.9309, 0.9346, 0.9488, 0.8814, 0.6857]
    }
    ml_bench_radar_chart(models)

    def analyze_agents(agents, avg_score, costs):
        plt.style.use('seaborn-v0_8')
        fig, ax1 = plt.subplots(figsize=(12, 8))  # 增大图形尺寸
        score_color = '#55A868'
        cost_color = '#4C72B0'

        # 设置x轴和y轴标签字体大小
        ax1.set_xlabel('数据科学智能体', fontproperties=font_prop, fontsize=24)  # 放大x轴标签字体
        ax1.set_ylabel('平均性能得分', fontproperties=font_prop, fontsize=24, color=score_color)  # 放大y轴标签字体
        ax1.tick_params(axis='y', labelcolor=score_color, labelsize=18)  # 放大y轴刻度字体
        ax1.set_ylim(0, max(avg_score) * 1.2)  # 动态调整y轴范围

        # 画平均分数柱状图
        x = np.arange(len(agents))
        bar_width = 0.35
        score_bars = ax1.bar(x - bar_width / 2, avg_score, bar_width, color=score_color, label='平均性能得分', alpha=0.8)
        # 为平均分数添加数值标签
        for bar in score_bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.02, f'{height:.4f}', ha='center', va='bottom',
                     fontsize=18, fontweight='bold', color=score_color)  # 放大数值标签字体

        # 创建右边y轴（成本）
        ax2 = ax1.twinx()
        ax2.set_ylabel('成本', fontproperties=font_prop, fontsize=24, color=cost_color)  # 放大y轴标签字体
        ax2.tick_params(axis='y', labelcolor=cost_color, labelsize=18)  # 放大y轴刻度字体
        ax2.set_ylim(0, 0.2)
        cost_bars = ax2.bar(x + bar_width / 2, costs, bar_width, color=cost_color, label='成本', alpha=0.8)
        # 为成本添加数值标签
        for bar in cost_bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.003, f'{height:.4f}', ha='center', va='bottom',
                     fontsize=18, fontweight='bold', color=cost_color)  # 放大数值标签字体

        # 设置x轴刻度和标签
        ax1.set_xticks(x)
        ax1.set_xticklabels(agents, fontproperties=font_prop, fontsize=18)  # 放大x轴刻度标签字体
        # 添加网格线
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', prop={'family': 'Microsoft YaHei', 'size': 18})
        plt.title('代理性能与成本对比', fontproperties=font_prop, fontsize=28, fontweight='bold')  # 放大标题字体
        plt.tight_layout()
        plt.show()  # 直接显示图表

    agents = ['AutoGen', 'DataInterpreter', 'W-LATS']
    avg_score = [0.5531, 0.7665, 0.9205]
    costs = [0.0347, 0.075, 0.092]
    analyze_agents(agents, avg_score, costs)


def w_lats_ablation():
    models = {
        '消融经验知识驱动': [1, 1, 0.9802, 0.9134, 0.9328, 0.9306, 0.6613, 0.5571],
        '消融双层级反思': [1, 1, 0.9825, 0.9148, 0.935, 0.9088, 0.9054, 0.6846],
        '完整方法': [1, 1, 0.9825, 0.9309, 0.9346, 0.9488, 0.9392, 0.6857],
        # 'W-LATS(w/o exp-driven)': [1, 1, 0.9802, 0.9134, 0.9328, 0.9306, 0.6613, 0.5571],
        # 'W-LATS': [1, 1, 0.9825, 0.9309, 0.9346, 0.9488, 0.9392, 0.6857]
    }
    ml_bench_radar_chart(models)
    agents = ['消融经验知识驱动', '消融双层级反思', '完整方法']
    costs = [0.164, 0.095, 0.079]
    cost_visualize(agents, costs)


if __name__ == '__main__':
    # w_lats_radar()
    w_lats_ablation()

