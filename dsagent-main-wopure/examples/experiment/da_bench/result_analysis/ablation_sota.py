import matplotlib.pyplot as plt

from examples.experiment.da_bench.result_analysis.data_visualize import format_accuracy_visualize

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from examples.experiment.da_bench.result_analysis.data_visualize import format_accuracy_visualize


def ablation_visualize():
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 数据
    # methods = ['DI', 'W-Plan', 'W-Plan（消融经验知识增量提取模块）', 'W-Plan（消融工作流检索模块）']
    methods = ['DI', 'W-Plan', 'W-Plan(w/o reflect_extractor)', 'W-Plan(w/o workflow_augment)']
    format_res = [[0.7009, 0.7502, 0.7623], [0.8249, 0.864, 0.8525], [0.8016, 0.8531, 0.8351], [0.7724, 0.8268, 0.8085]]
    format_accuracy_visualize(format_res, methods, "")
    wo_format_success_rates = [57.39, 80.16, 77.24, 73.92]
    formated_success_rates = [70.09, 82.49, 80.16, 77.24]
    bubble_sizes = [20, 50, 30, 30]  # 调整气泡大小
    colors = ['red', 'purple', 'blue', 'pink']

    # 绘制图形
    plt.figure(figsize=(8.5, 4.5))
    for i in range(len(methods)):
        plt.scatter(formated_success_rates[i], [0], s=bubble_sizes[i] ** 2, color=colors[i], alpha=0.6,
                    label=methods[i])

    # 添加注释（标注成功率）
    # for i, rate in enumerate(formated_success_rates):
    #     plt.text(rate, -0.01, f"{rate}%", ha='center', fontsize=10)

    # 自定义图形样式
    plt.axhline(y=0, color='black', linewidth=0.5)  # 横线
    plt.yticks([])  # 隐藏 y 轴刻度
    plt.xticks(range(60, 100, 5), fontsize=14)  # 增大刻度字体
    # plt.xlabel('准确率 (%)', fontsize=16)  # 更换为中文标签
    # plt.title('消融研究', fontsize=18)  # 更换为中文标题
    plt.xlabel('Accuracy %', fontsize=16)  # 增大标签字体
    plt.title('Ablation Study', fontsize=18)  # 增大标题字体

    handles, labels = plt.gca().get_legend_handles_labels()
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=colors[i],
                                 markersize=12,  # 增大标记尺寸
                                 linestyle='') for i in range(len(labels))]

    plt.legend(handles=legend_handles,
               labels=labels,
               ncol=2,  # 分两列展示
               loc='upper left',
               bbox_to_anchor=(0.02, 0.98),  # 调整位置到左上角边缘
               fontsize=18,  # 增大字体
               frameon=True,
               framealpha=0.9,
               edgecolor='black',
               handletextpad=0.5,  # 调整文本与标记的间距
               columnspacing=1.2)  # 调整列间距

    plt.tight_layout(pad=2)  # 增加布局边距
    plt.savefig('消融实验.png', dpi=300, bbox_inches='tight')  # 保存图片
    plt.show()


def sota_visualize():
    methods = ['Infiagent(gpt-4o-2024-08-06)', 'our (gpt-4o-mini-2024-07-18)']
    success_rates = [72.56, 82.49]
    costs = [4.14, 0.2955]  # 单位是 $
    bubble_sizes = [25, 50]
    colors = ['pink', 'purple']

    # 创建一个2行1列的布局
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [2, 1]})

    # 上方气泡图
    for i in range(len(methods)):
        axs[0].scatter(success_rates[i], [0], s=bubble_sizes[i] ** 2, color=colors[i], alpha=0.6, label=methods[i])
    axs[0].axhline(y=0, color='black', linewidth=0.5)  # 横线
    axs[0].set_yticks([])  # 隐藏 y 轴刻度
    axs[0].set_xticks(range(50, 100, 5))  # 自定义 x 轴刻度
    plt.setp(axs[0].get_xticklabels(), fontsize=10)  # 设置 x 轴刻度字体大小
    axs[0].set_xlabel('success rate %', fontsize=12)
    axs[0].set_title('SOTA LLM Study', fontsize=14)

    # 自定义图例句柄大小并放入图内
    handles, labels = axs[0].get_legend_handles_labels()
    legend_handles = [plt.Line2D([0], [-20], marker='o', color='w', markerfacecolor=colors[i],
                                 markersize=5) for i in range(len(labels))]
    axs[0].legend(legend_handles, labels, title='', loc='upper left', bbox_to_anchor=(0.05, 0.9), fontsize=10,
                  frameon=True, framealpha=0.9, edgecolor='black')

    # 下方柱状图
    bars = axs[1].bar(methods, costs, color=colors, alpha=0.7)
    axs[1].set_ylabel('Cost in $', fontsize=12)
    axs[1].set_title('Cost Comparison', fontsize=14)
    axs[1].set_ylim(0, max(costs) * 1.2)  # 设置 y 轴的范围，稍微加大上限

    for bar in bars:
        yval = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f"${yval:.2f}", ha='center', fontsize=10)

    # 自动调整布局
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    ablation_visualize()
