import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches


def plot_weight_transition_ds(save_path='weight_transition_ds.png', dpi=300):
    """
    生成深度感知权重迁移示意图，带斜线背景和中文标签
    :param save_path: 图片保存路径
    :param dpi: 输出分辨率
    :return: 图片保存路径
    """
    # 配置全局样式
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'Microsoft YaHei'  # 设置全局字体为微软雅黑
    plt.rcParams['axes.labelsize'] = 14  # 增大标签字体
    plt.rcParams['legend.fontsize'] = 12  # 增大图例字体
    plt.rcParams['xtick.labelsize'] = 12  # 增大刻度字体
    plt.rcParams['ytick.labelsize'] = 12  # 增大刻度字体

    # 生成数据
    depths = np.linspace(0, 10, 100)
    goal_weights = 0.2 + 0.3 * (depths / 10)
    code_weights = 0.3 + 0.1 * (depths / 10)
    prospect_weights = 0.5 - 0.4 * (depths / 10)

    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    # 创建斜线图案填充 - 使用更细的线条
    exploration_pattern = mpatches.Patch(
        facecolor='#e6f3ff',
        hatch='///',
        alpha=0.2,  # 降低透明度
        label='探索阶段'
    )
    balanced_pattern = mpatches.Patch(
        facecolor='#fff2e6',
        hatch='..',  # 使用更细的点状图案
        alpha=0.2,
        label='平衡阶段'
    )
    exploitation_pattern = mpatches.Patch(
        facecolor='#e6ffe6',
        hatch='\\\\\\',
        alpha=0.2,  # 降低透明度
        label='开发阶段'
    )

    # 添加斜线背景 - 使用更细的线条和更低的透明度
    ax.axvspan(0, 3, facecolor='#e6f3ff', alpha=0.2, hatch='///', label='探索阶段')
    ax.axvspan(3, 7, facecolor='#fff2e6', alpha=0.2, hatch='..', label='平衡阶段')  # 更改为点状
    ax.axvspan(7, 10, facecolor='#e6ffe6', alpha=0.2, hatch='\\\\\\', label='开发阶段')

    # 绘制核心曲线 - 增加线宽
    ax.plot(depths, goal_weights, color='#1f77b4', lw=4, label='目标完成度')  # 增加线宽
    ax.plot(depths, code_weights, color='#ff7f0e', lw=4, label='代码质量')  # 增加线宽
    ax.plot(depths, prospect_weights, color='#2ca02c', lw=4, label='前景价值')  # 增加线宽

    # 设置坐标轴
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 0.6)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel('树搜索深度', fontsize=14, labelpad=10)
    ax.set_ylabel('权重系数', fontsize=14, labelpad=10)

    # 添加中文注释
    ax.text(1.5, 0.62, '注重探索策略', ha='center', fontsize=12, color='#1f77b4', weight='bold')
    ax.text(8.5, 0.62, '注重目标完成', ha='center', fontsize=12, color='#1f77b4', weight='bold')

    # 优化"动态权重迁移"箭头 - 加粗箭头并调整样式
    ax.annotate('动态权重迁移', xy=(5, 0.42), xytext=(3, 0.61),
                arrowprops=dict(
                    arrowstyle="fancy",  # 使用更明显的箭头样式
                    color='black',  # 更改为黑色以增加对比度
                    lw=2.5,  # 增加线宽
                    connectionstyle="arc3,rad=.2",  # 添加弧度
                    shrinkA=5,
                    shrinkB=5
                ),
                fontsize=13, weight='bold', color='black')  # 文字加粗

    # 增强可读性
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=5)
    ax.grid(True, alpha=0.3, ls='--')  # 降低网格线透明度

    # 组合图例 - 使用自定义的patch对象
    lines = ax.get_lines()
    handles = [exploration_pattern, balanced_pattern, exploitation_pattern,
               lines[0], lines[1], lines[2]]
    labels = ['探索阶段', '平衡阶段', '开发阶段',
              '目标完成度', '代码质量', '前景价值']

    ax.legend(handles, labels, loc='upper center',
              bbox_to_anchor=(0.5, -0.15), ncol=3,
              fontsize=12, frameon=True)

    # 保存输出
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close()

    return save_path


def plot_weight_transition(save_path: str = None, dpi: int = 300):
    """
    可视化深度权重迁移曲线，带斜线背景和中文标签

    参数：
    save_path: 图片保存路径（None时显示窗口）
    dpi: 输出分辨率（默认300）
    """
    # 配置学术图表样式
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'Microsoft YaHei',  # 设置中文字体
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelpad': 10,
        'xtick.major.pad': 8,
        'ytick.major.pad': 8,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

    # 生成数据
    depths = np.linspace(0, 10, 100)
    goal_weights = 0.3 + 0.2 * depths / 10
    code_weights = 0.3 + 0.1 * depths / 10
    prospect_weights = 0.4 - 0.3 * depths / 10

    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 5))

    # 添加斜线填充区域
    ax.axvspan(0, 3.5, facecolor='#e6f3ff', alpha=0.2, hatch='///', label='探索区域')
    ax.axvspan(3.5, 7, facecolor='#fff2e6', alpha=0.2, hatch='xxx', label='平衡区域')
    ax.axvspan(7, 10, facecolor='#e6ffe6', alpha=0.2, hatch='\\\\\\', label='开发区域')

    # 绘制渐变曲线
    line_goal, = ax.plot(depths, goal_weights,
                         color='#1f77b4', linewidth=3,
                         linestyle='-', marker='',
                         label='目标权重')
    line_code, = ax.plot(depths, code_weights,
                         color='#ff7f0e', linewidth=3,
                         linestyle=(0, (5, 3)),
                         label='代码权重')
    line_prospect, = ax.plot(depths, prospect_weights,
                             color='#2ca02c', linewidth=3,
                             linestyle='-.',
                             label='前景权重')

    # 设置坐标轴
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(0, 0.55)
    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_yticks(np.linspace(0, 0.5, 6))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    # 添加标注
    ax.set_xlabel('搜索深度 (d)', fontweight='bold', fontsize=14)
    ax.set_ylabel('权重值', fontweight='bold', fontsize=14)
    ax.set_title('搜索深度下的动态权重迁移',
                 fontweight='bold', pad=15, fontsize=16)

    # 添加区域标注
    ax.text(1.5, 0.52, '探索优先', ha='center', fontsize=12, weight='bold')
    ax.text(5, 0.52, '权重平衡', ha='center', fontsize=12, weight='bold')
    ax.text(8.5, 0.52, '目标优先', ha='center', fontsize=12, weight='bold')

    # 创建自定义图例元素
    exploration_pattern = mpatches.Patch(facecolor='#e6f3ff', hatch='///', alpha=0.2, label='探索区域')
    balanced_pattern = mpatches.Patch(facecolor='#fff2e6', hatch='xxx', alpha=0.2, label='平衡区域')
    exploitation_pattern = mpatches.Patch(facecolor='#e6ffe6', hatch='\\\\\\', alpha=0.2, label='开发区域')

    # 增强图例
    handles = [line_goal, line_code, line_prospect,
               exploration_pattern, balanced_pattern, exploitation_pattern]
    labels = ['目标权重', '代码权重', '前景权重',
              '探索区域', '平衡区域', '开发区域']

    legend = ax.legend(handles, labels,
                       loc='upper center',
                       bbox_to_anchor=(0.5, -0.2),
                       ncol=3,
                       fontsize=12,
                       title='权重组件与搜索区域:',
                       title_fontsize=14,
                       frameon=True)
    legend.get_frame().set_facecolor('#f5f5f5')

    # 添加辅助线
    for d in [0, 5, 10]:
        ax.axvline(d, color='gray', linestyle=':',
                   alpha=0.6, linewidth=1.2)

    # 优化布局
    plt.tight_layout()

    # 输出控制
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white')
        print(f"图表已保存至 {save_path}")
    else:
        plt.show()

    plt.close()


# 使用示例
plot_weight_transition_ds()

