import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置脚本所在目录为基础路径
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, 'avg_finetune-moe.csv')

# 读取 CSV 数据
df = pd.read_csv(file_path)

sns.set_theme(style="white", rc={"axes.facecolor": "white"})

# 方法顺序和颜色设置
methods = ['Federated Global Model', 'Local Expert Model', 'MoE Model']
		

colors = ["#99C64D", "#59B273", "#337581"]  # 自定义颜色

# 获取所有 alpha 值（按顺序去重）
alpha_values = sorted(df['alpha'].unique())

for alpha in alpha_values:
    # 筛选当前 alpha 的数据
    df_alpha = df[df['alpha'] == alpha].copy()

    # 重构 DataFrame 以便于绘图
    df_melted = df_alpha.melt(
        id_vars='out_ratio',
        value_vars=methods,
        var_name='Method',
        value_name='Accuracy'
    )

    legend_name_map = {
        'Federated Global Model': r'Federated Global Model $F_g$',
        'Local Expert Model': r'Local Specialized Model $F_s$',
        'MoE Model': r'MoE Model $F_{MoE}$'
    }
    df_melted['Method'] = df_melted['Method'].map(legend_name_map)

    # 创建图形（适当加高）
    plt.figure(figsize=(8, 7))
    ax = sns.barplot(
        data=df_melted,
        x='out_ratio',
        y='Accuracy',
        hue='Method',
        palette=colors,
        saturation=1,
        width=0.8
    )

    # 为每个条形添加数值标签
    for container in ax.containers:
        ax.bar_label(container, padding=3, fmt='%.4f', fontsize=9)


    # 标题和轴标签
    plt.title(f'Local Validation/Accuracy (alpha={alpha})', fontsize=13)
    plt.xlabel('out_ratio', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)

    # 设置 Y 轴范围为 0~1.2，但只显示到 1.0
    plt.ylim(0, 1.2)
    plt.yticks(np.arange(0, 1.01, 0.1))  # 显示刻度只到 1.0

    # 图例固定位置
    plt.legend(loc='upper left')

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 保证顶部空间足够

    # 保存为图像
    save_path = os.path.join(base_dir, f'accuracy_comparison_alpha_{alpha}.png')
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()