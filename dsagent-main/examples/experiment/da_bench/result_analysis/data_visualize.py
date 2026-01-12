import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm


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


set_chinese_font()


def overall_bar_chart():
    # 数据
    agents = ['task_weaver', 'DI', 'our']
    values1 = [0.3074, 0.5739, 0.7724]
    values2 = [0.4276, 0.6359, 0.807]
    values3 = [0.3395, 0.6398, 0.7978]
    colors = ['#FFBE7A', '#FA7F6F', '#82B0D2']

    # 设置位置和宽度
    bar_width = 0.2
    index = np.arange(len(agents))

    plt.figure(figsize=(10, 6))

    # 绘制柱状图
    bars1 = plt.bar(index, values1, bar_width, label='accuracy_by_question', edgecolor='black', color="#FADCAA")
    bars2 = plt.bar(index + bar_width, values2, bar_width, label='accuracy_by_sub_question', edgecolor='black',
                    color="#A6D0DD")
    bars3 = plt.bar(index + 2 * bar_width, values3, bar_width, label='proportional_accuracy_by_sub_question',
                    edgecolor='black', color="#82A0D8")

    # 设置Y轴范围
    plt.ylim(0, 1)

    # 添加轴标签和图例
    plt.xlabel('Agents', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(index + 1 * bar_width, agents, fontsize=12)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=12)

    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height()}', ha='center', va='bottom')
    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height()}', ha='center', va='bottom')
    for bar in bars3:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height()}', ha='center', va='bottom')

    plt.show()


def format_accuracy_visualize(format_res: list, agents: list, title: str, colors: list = None):
    metrics = ['整体准确性', '子问题准确性', '加权准确性']
    # 设置默认学术配色（浅色系）
    if colors is None:
        colors = [
             '#4C72B0',  # 柔和的蓝色
             '#55A868',  # 学术绿
             '#C44E52',  # 砖红色
             '#8172B3',  # 淡紫色
             '#CCB974',  # 卡其色
             '#64B5CD',  # 天蓝色
             '#D65F5F',  # 珊瑚红
             '#6C7A89',  # 石板灰
             '#7A68A6',  # 深紫色
             '#A9A9A9',  # 暗灰色
             '#F0E442',  # 明亮的黄色
             '#D55E00',  # 橙色
             '#009E73',  # 深绿色
             '#CC79A7',  # 粉紫色
             '#0072B2',  # 深蓝色
             '#E69F00',  # 琥珀色
             '#56B4E9',  # 天空蓝
             '#2C3E50',  # 深青色
             '#E74C3C',  # 朱红色
             '#9B59B6'  # 紫水晶色
         ][:len(agents)]  # 自动截取所需长度

    # 设置图表的宽度，根据agents数量调整
    width = 0.8 / len(agents)
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, agent in enumerate(agents):
        agent_accuracies = format_res[i]
        x = np.arange(len(metrics))
        pos = x - 0.4 + width * (i + 0.5)

        bars = ax.bar(pos, agent_accuracies, width,
                      label=agent, color=colors[i])
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points", size=15,
                        ha='center', va='bottom')

    ax.set_ylabel('准确性', fontproperties='Microsoft YaHei', fontsize=20)
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontproperties='Microsoft YaHei', fontsize=20)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='y', labelsize=18)  # 设置y轴刻度字体大小

    # 优化图例显示
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),  ncol=len(agents) if len(agents) <= 4 else 4, frameon=False, prop={'family': 'Microsoft YaHei', 'size': 20})
    ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.set_facecolor('#F9F9F9')
    fig.patch.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig, ax


def unformat_vs_format_visualize(unformat_res: list, format_res: list, agents: list, title: str):
    bar_width = 0.35
    index = np.arange(len(agents))

    bars1 = plt.bar(index, unformat_res, bar_width, label='wo/reformat', color="#1f77b4", edgecolor='black')
    bars2 = plt.bar(index + bar_width, format_res, bar_width, label='w/reformat', color="#ff7f0e", edgecolor='black')

    plt.title(f'{title}', fontsize=16)
    plt.xlabel('agents', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.xticks(index + bar_width / 2, agents)
    # plt.ylim(0, 1)
    plt.legend(fontsize=15)
    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height()}', ha='center', va='bottom')
    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height()}', ha='center', va='bottom')
    plt.show()


def token_cost_visualize(agents: list, token_cost: list, title: str = "token_cost"):
    # colors = ['#FFBE7A', '#FFBE7A', '#FFBE7A' '#82B0D2']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    plt.bar(agents, token_cost, color=colors)
    plt.title(f'{title}', fontsize=16)
    plt.xlabel('agents', fontsize=12)
    plt.ylabel('token_cost', fontsize=12)
    for i in range(len(token_cost)):
        plt.text(i, token_cost[i] + 0.1, str(token_cost[i]), ha='center')
    plt.show()


def line_chart_visualize(a_values: list, b_values: list, c_values: list, d_values: list, title: str = "准确率对比"):
    x_labels = ["整体准确率", "子问题准确率", "加权准确率"]
    x_positions = range(len(x_labels))
    plt.figure(figsize=(8, 6))
    plt.plot(x_positions, a_values, marker='o', label="AutoGen", color='blue')
    plt.plot(x_positions, b_values, marker='o', label="TaskWeaver", color='green')
    plt.plot(x_positions, c_values, marker='o', label="DataInterpreter", color='red')
    plt.plot(x_positions, d_values, marker='o', label="本文方法", color='purple')
    for y_values, color in zip([a_values, b_values, c_values, d_values], ['blue', 'green', 'red', 'purple']):
        for x, y in zip(x_positions, y_values):
            plt.text(x, y, f"{y}", fontsize=14, color=color, ha='center', va='bottom')
    plt.xticks(x_positions, x_labels, fontsize=14)
    plt.title(title, fontsize=20)
    plt.xlabel("评估指标", fontsize=18)  # 改为中文
    plt.ylabel("准确率", fontsize=18)  # 改为中文
    plt.ylim(0, 1)
    plt.legend(fontsize=15)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def hard_task_linear_visualize():
    format_hard_autogen_gpt, format_hard_task_weaver_gpt, format_hard_di_gpt, format_hard_our_gpt = (
        [0.3977, 0.4409, 0.4479], [0.4545, 0.4975, 0.5057], [0.6193, 0.6995, 0.6983], [0.7557, 0.8325, 0.8082]
    )
    format_hard_hard_glm, format_hard_task_weaver_glm, format_hard_di_glm, format_hard_our_glm = (
        [0.2727, 0.2808, 0.3447], [0.1079, 0.1404, 0.1358], [0.4319, 0.5567, 0.5256], [0.4489, 0.5691, 0.5606]
    )
    unformat_hard_autogen_gpt, unformat_hard_task_weaver_gpt, unformat_hard_di_gpt, unformat_hard_our_gpt = (
        [0.0114, 0.069, 0.053], [0.3806, 0.4778, 0.4372], [0.4489, 0.5443, 0.5492], [0.7102, 0.7981, 0.7685]
    )
    unformat_hard_hard_glm, unformat_hard_task_weaver_glm, unformat_hard_di_glm, unformat_hard_our_glm = (
        [0.0114, 0.0049, 0.0114], [0.0512, 0.0616, 0.061], [0.3012, 0.4137, 0.3822], [0.4205, 0.532, 0.5247]
    )
    format_hard_lats_gpt, format_hard_lats_glm = [0.7898, 0.8498, 0.841], [0.5795, 0.6601, 0.6676]
    line_chart_visualize(format_hard_autogen_gpt, format_hard_task_weaver_gpt, format_hard_di_gpt, format_hard_our_gpt, "gpt_4o_mini 标准化结果")
    line_chart_visualize(format_hard_hard_glm, format_hard_task_weaver_glm, format_hard_di_glm, format_hard_our_glm, "glm_4_flash 标准化结果")
    line_chart_visualize(unformat_hard_autogen_gpt, unformat_hard_task_weaver_gpt, unformat_hard_di_gpt, unformat_hard_our_gpt, "gpt_4o_mini w/o reformat")
    line_chart_visualize(unformat_hard_hard_glm, unformat_hard_task_weaver_glm, unformat_hard_di_glm, unformat_hard_our_glm, "glm_4_flash w/o reformat")
    line_chart_visualize(format_hard_autogen_gpt, format_hard_task_weaver_gpt, format_hard_di_gpt, format_hard_lats_gpt, "W-LATS gpt_4o_mini 标准化结果")
    line_chart_visualize(format_hard_hard_glm, format_hard_task_weaver_glm, format_hard_di_glm, format_hard_lats_glm, "W-LATS glm_4_flash 标准化结果")


def gpt_4o_mini_visualize():
    agents = ['autogen', 'task_weaver', 'DI', 'our']
    unformat_accuracy_by_question, format_accuracy_by_question = (
        [0.0039, 0.3171, 0.5739, 0.7724], [0.5486, 0.6109, 0.7009, 0.8249]
    )
    unformat_accuracy_by_sub_question, format_accuracy_by_sub_question = (
        [0.0724, 0.4616, 0.6359, 0.807], [0.5855, 0.6469, 0.7502, 0.864]
    )
    unformat_weighted_accuracy_by_sub_question, format_weighted_accuracy_by_sub_question = (
        [0.0496, 0.3802, 0.6398, 0.7978], [0.57, 0.6397, 0.7623, 0.8525]
    )

    unformat_vs_format_visualize(unformat_accuracy_by_question, format_accuracy_by_question, agents,
                                 'accuracy_by_question')
    unformat_vs_format_visualize(unformat_accuracy_by_sub_question, format_accuracy_by_sub_question, agents,
                                 'accuracy_by_sub_question')
    unformat_vs_format_visualize(unformat_weighted_accuracy_by_sub_question, format_weighted_accuracy_by_sub_question,
                                 agents, 'weighted_accuracy_by_sub_question')

    token_agents = ['DI', 'our']
    token_cost = [811702, 6860000, 1493196, 1742312]
    token_cost_visualize(agents, token_cost, 'token_cost')


def glm_4_flash_visualize():
    agents = ['autogen', 'task_weaver', 'DI', 'our']
    unformat_accuracy_by_question, format_accuracy_by_question = (
        [0.037, 0.1887, 0.5, 0.605], [0.3989, 0.3035, 0.6148, 0.6609]
    )
    unformat_accuracy_by_sub_question, format_accuracy_by_sub_question = (
        [0.0351, 0.1842, 0.534, 0.637], [0.4145, 0.3005, 0.6441, 0.676]
    )
    unformat_weighted_accuracy_by_sub_question, format_weighted_accuracy_by_sub_question = (
        [0.0454, 0.2066, 0.5404, 0.6487], [0.4405, 0.3215, 0.6404, 0.6962]
    )
    unformat_vs_format_visualize(unformat_accuracy_by_question, format_accuracy_by_question, agents,
                                 'accuracy_by_question')
    unformat_vs_format_visualize(unformat_accuracy_by_sub_question, format_accuracy_by_sub_question, agents,
                                 'accuracy_by_sub_question')
    unformat_vs_format_visualize(unformat_weighted_accuracy_by_sub_question, format_weighted_accuracy_by_sub_question,
                                 agents, 'weighted_accuracy_by_sub_question')
    token_cost = [635041, 8245601, 2389200, 3073823]
    token_cost_visualize(agents, token_cost, 'token_cost')


def lats_format_visualize():
    agents = ['Autogen', 'Taskweaver', 'DI', 'our']
    format_accuracy_4o_mini = [[0.5486, 0.5855, 0.57], [0.6109, 0.6469, 0.6397], [0.7009, 0.7502, 0.7623], [0.895, 0.909, 0.9135]]
    format_accuracy_4_flashx = [[0.3989, 0.4125, 0.4405], [0.3035, 0.3005, 0.3215], [0.6148, 0.6441, 0.6404], [0.7315, 0.7719, 0.7711]]
    format_accuracy_visualize(format_accuracy_4o_mini, agents, "")
    format_accuracy_visualize(format_accuracy_4_flashx, agents, "")


def w_plan_format_visualize():
    agents = ['Autogen', 'Taskweaver', 'DI', 'our']
    format_accuracy_4o_mini = [[0.5486, 0.5855, 0.57], [0.6109, 0.6469, 0.6397], [0.7009, 0.7502, 0.7623], [0.8249, 0.864, 0.8525]]
    format_accuracy_4_flashx = [[0.3989, 0.4125, 0.4405], [0.3035, 0.3005, 0.3215], [0.6148, 0.6441, 0.6404], [0.6609, 0.676, 0.6962]]
    format_accuracy_visualize(format_accuracy_4o_mini, agents, "")
    format_accuracy_visualize(format_accuracy_4_flashx, agents, "")


def main():
    pass
    # gpt_4o_mini_visualize()
    # glm_4_flash_visualize()
    hard_task_linear_visualize()     # hard任务 可视化折线图
    # our_vs_gpt_4o_visualize()
    # lats_format_visualize()         # lats方法 格式化准确性对比
    # w_plan_format_visualize()         # w-plan方法 格式化准确性对比


if __name__ == '__main__':
    main()
