import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import os
import matplotlib.font_manager as fm

from examples.experiment.da_bench.result_analysis.data_visualize import format_accuracy_visualize


# 函数：设置中文字体
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

# 创建数据框
data = {
    '基础模型': ['GPT-4o', 'GPT-4o', 'GPT-4o', 'DeepSeek-R1', 'o3-mini', 'o1', 'GPT-4o-mini'],
    '智能体': ['ReAct', 'DataInterpreter', '本文方法', 'ReAct', 'ReAct', 'ReAct', '本文方法'],
    '整体准确率': [0.7899, 0.7123, 0.8678, 0.7432, 0.7977, 0.8210, 0.8249],
    '子问题准确率': [0.8491, 0.7456, 0.8992, 0.7368, 0.8531, 0.8596, 0.8640],
    '加权准确率': [0.8448, 0.7632, 0.8972, 0.7876, 0.8351, 0.8468, 0.8525],
    'Cost($)': [4.14, 2.75, 3.96, 3.69, 10.58, 100.24, 0.314]
}

df = pd.DataFrame(data)


# def format_accuracy_comparison():
#     agents = ['GPT-4o_ReAct', 'GPT-4o_DataInterpreter', 'DeepSeek-R1_ReAct', 'o3-mini_ReAct', 'o1_ReAct', 'GPT-4o-mini_本文方法', 'GPT-4o_本文方法']
#     format_res = [[0.7899, 0.8491, 0.8448], [0.7123, 0.7456, 0.7632], [0.7432, 0.7368, 0.7876], [0.7977, 0.8531, 0.8351], [0.8210, 0.8596, 0.8468], [0.8249, 0.8640, 0.8525], [0.8678, 0.8992, 0.8972]]
#     format_accuracy_visualize(format_res, agents, "")


def plot_accuracy_comparison(df, font_prop=None):
    """
    绘制准确率对比图 - 所有模型，并为每个柱子添加具体数值
    """
    plt.figure(figsize=(8, 6))
    metrics = ['整体准确率', '子问题准确率', '加权准确率']
    df['模型_智能体'] = df['基础模型'] + ' - ' + df['智能体']
    df_long = pd.melt(df, id_vars=['模型_智能体'], value_vars=metrics,
                      var_name='评估指标', value_name='准确率')
    order = df.sort_values(by='加权准确率', ascending=False)['模型_智能体'].tolist()

    ax = sns.barplot(x='模型_智能体', y='准确率', hue='评估指标', data=df_long, order=order)

    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=9, padding=3)  # 格式化显示小数点后三位

    if font_prop:
        plt.title('各模型在不同评估指标上的准确率对比', fontproperties=font_prop, fontsize=14)
        plt.xlabel('模型与智能体', fontproperties=font_prop, fontsize=12)
        plt.ylabel('准确率', fontproperties=font_prop, fontsize=12)
        plt.xticks(rotation=45, ha='right', fontproperties=font_prop)
        legend = plt.legend(title='评估指标')
        plt.setp(legend.get_title(), fontproperties=font_prop)
        for text in legend.get_texts():
            plt.setp(text, fontproperties=font_prop)
    else:
        plt.title('各模型在不同评估指标上的准确率对比', fontsize=14)
        plt.xlabel('模型与智能体', fontsize=12)
        plt.ylabel('准确率', fontsize=12)
        plt.xticks(rotation=45, ha='right')

    plt.ylim(0.7, 0.95)
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_cost_vs_accuracy(df, font_prop=None):
    """
    绘制成本效益分析 - 加权准确率与成本的关系
    """
    plt.figure(figsize=(8, 6))  # 增加图像大小
    ax = plt.gca()

    scatter = ax.scatter(df['Cost($)'], df['加权准确率'],
                         s=df['加权准确率'] * 1000,  # 增加点的大小
                         alpha=0.9,
                         c=['#3498db' if x != '本文方法' else '#e74c3c' for x in df['智能体']])

    for i, row in df.iterrows():
        label = f"{row['基础模型']}\n{row['智能体']}"
        if font_prop:
            ax.annotate(label, (row['Cost($)'], row['加权准确率']),
                        xytext=(15, -15), textcoords='offset points',
                        fontproperties=font_prop, fontsize=17)
        else:
            ax.annotate(label, (row['Cost($)'], row['加权准确率']),
                        xytext=(7, 5), textcoords='offset points',
                        fontsize=17)  #

    if font_prop:
        plt.title('成本与加权准确率的关系', fontproperties=font_prop, fontsize=18)
        plt.xlabel('成本 ($)', fontproperties=font_prop, fontsize=16)
        plt.ylabel('加权准确率', fontproperties=font_prop, fontsize=16)
        plt.xticks(fontproperties=font_prop, fontsize=14)
        plt.yticks(fontproperties=font_prop, fontsize=14)
    else:
        plt.title('成本与加权准确率的关系', fontsize=18)
        plt.xlabel('成本 ($)', fontsize=16)
        plt.ylabel('加权准确率', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

    plt.xscale('log')
    plt.grid(True, alpha=0.3)

    # 添加图例以区分不同点的颜色含义
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
               markersize=14, label='其他方法', alpha=0.9),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
               markersize=14, label='本文方法', alpha=0.9)
    ]
    if font_prop:
        # 创建一个新的字体属性对象，复制原来的属性但设置新的大小
        legend_font = font_prop.copy()
        legend_font.set_size(16)  # 设置图例字体大小
        ax.legend(handles=legend_elements, prop=legend_font, loc='upper right')
    else:
        ax.legend(handles=legend_elements, fontsize=16, loc='upper right')

    plt.tight_layout()
    plt.savefig('cost_vs_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_cost_efficiency():
    """
    可视化成本效益分析
    """
    plt.figure(figsize=(14, 6))

    # 计算成本效益比 (加权准确率/成本)
    df['效益比'] = df['加权准确率'] / df['Cost($)']

    # 按效益比排序
    df_sorted = df.sort_values('效益比', ascending=False)

    # 设置颜色
    colors = ['#e74c3c' if x == '本文方法' else '#3498db' for x in df_sorted['智能体']]

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 6))

    # 绘制条形图，修复FutureWarning
    bars = ax.bar(df_sorted['模型_智能体'], df_sorted['效益比'],
                  color=colors)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        if font_prop:
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontproperties=font_prop)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')

    if font_prop:
        plt.title('各模型成本效益比对比 (加权准确率/成本)', fontproperties=font_prop, fontsize=14)
        plt.xlabel('模型-智能体', fontproperties=font_prop, fontsize=12)
        plt.ylabel('效益比 (加权准确率/成本)', fontproperties=font_prop, fontsize=12)
        plt.xticks(rotation=45, ha='right', fontproperties=font_prop)
    else:
        plt.title('各模型成本效益比对比 (加权准确率/成本)', fontsize=14)
        plt.xlabel('模型-智能体', fontsize=12)
        plt.ylabel('效益比 (加权准确率/成本)', fontsize=12)
        plt.xticks(rotation=45, ha='right')

    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('cost_efficiency_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_radar_chart():
    """
    创建雷达图比较不同方法
    """
    # 选择GPT-4o的三种方法进行比较
    radar_df = df[df['基础模型'] == 'GPT-4o']

    # 准备雷达图数据
    categories = ['整体准确率', '子问题准确率', '加权准确率']

    # 创建角度
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    # 设置图形
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 为每个方法添加数据
    for i, row in radar_df.iterrows():
        values = [row['整体准确率'], row['子问题准确率'], row['加权准确率']]
        values += values[:1]  # 闭合数据

        color = '#e74c3c' if row['智能体'] == '本文方法' else '#3498db'
        ax.plot(angles, values, 'o-', linewidth=2, label=row['智能体'], color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    # 设置角度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([str(cat) for cat in categories], fontproperties=font_prop, fontsize=10)

    # 设置Y轴范围
    ax.set_ylim(0.7, 0.95)

    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), prop=font_prop)
    plt.title('GPT-4o上不同方法的性能雷达图', fontproperties=font_prop, y=1.1, fontsize=14)

    plt.tight_layout()
    plt.savefig('radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_bubble_chart():
    """
    创建高级气泡图，展示三个维度的性能指标
    """
    plt.figure(figsize=(12, 8))

    # 设置气泡大小和颜色
    sizes = df['整体准确率'] * 1000  # 气泡大小基于整体准确率
    colors = ['#e74c3c' if x == '本文方法' else '#3498db' for x in df['智能体']]

    # 创建散点图
    scatter = plt.scatter(df['加权准确率'], df['子问题准确率'],
                          s=sizes, c=colors, alpha=0.7, edgecolors='w')

    # 添加标签
    for i, row in df.iterrows():
        plt.annotate(f"{row['基础模型']}-{row['智能体']}\nCost: ${row['Cost($)']}",
                     (row['加权准确率'], row['子问题准确率']),
                     xytext=(7, 0), textcoords='offset points', fontproperties=font_prop, fontsize=9)

    # 设置图表属性
    plt.title('模型性能多维度对比分析', fontproperties=font_prop, fontsize=16)
    plt.xlabel('加权准确率', fontproperties=font_prop, fontsize=12)
    plt.ylabel('子问题准确率', fontproperties=font_prop, fontsize=12)

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)

    # 创建图例说明气泡大小代表整体准确率
    bubble_sizes = [0.75, 0.80, 0.85]
    legend_bubbles = []
    for size in bubble_sizes:
        legend_bubbles.append(plt.scatter([], [], s=size * 1000, c='gray', alpha=0.7, edgecolors='w'))

    labels = [f'整体准确率: {s}' for s in bubble_sizes]
    plt.legend(legend_bubbles, labels, loc='upper left', frameon=True, prop=font_prop, title='图例',
               title_fontproperties=font_prop)

    # 添加参考线
    plt.axhline(y=df['子问题准确率'].mean(), color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=df['加权准确率'].mean(), color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('performance_bubble_chart.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_model_performance(df, font_prop=None):
    """
    调用所有独立的绘图方法以生成完整的性能分析报告
    """
    sns.set(style="whitegrid")

    # 调用独立的绘图方法
    # format_accuracy_comparison()
    # plot_accuracy_comparison(df, font_prop)
    plot_cost_vs_accuracy(df, font_prop)

    # visualize_cost_efficiency()
    # create_radar_chart()
    # create_bubble_chart()


if __name__ == '__main__':
    # 执行可视化函数
    visualize_model_performance(df, font_prop=font_prop)
