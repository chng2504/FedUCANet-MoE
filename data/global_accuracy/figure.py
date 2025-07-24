import os
import pandas as pd
import matplotlib.pyplot as plt

# 设置工作目录为脚本所在目录
base_dir = os.path.dirname(os.path.abspath(__file__))

# 所有的 alpha 和 out_ratio 取值
alphas = ['0.1', '0.5', '1', '10']
out_ratios = ['0.0', '0.2', '0.4', '0.6']
colors = ['#274753', '#299d8f', '#e7c66b', '#e66d50']
markers = ['o', 's', '^', 'D']  # 圆圈、正方形、三角、菱形

for alpha in alphas:
    plt.figure(figsize=(8, 6))
    for i, out_ratio in enumerate(out_ratios):
        file_path = os.path.join(base_dir, f"alpha={alpha}", f"out_ratio({out_ratio}).csv")

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        plt.plot(
            df['step'],
            df['global/validate-acc'],
            label=f'out_ratio={out_ratio}',
            color=colors[i],
            marker=markers[i],
            linestyle='-',
            markersize=6
        )

    plt.title(f'Global Validation Accuracy (alpha={alpha})')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy')


    x_ticks = list(range(20))
    x_labels = [str(i) for i in range(20)]  
    plt.xticks(ticks=x_ticks, labels=x_labels)

    # 设置 y 轴从 0.0 到 1.0，间隔 0.1
    y_ticks = [round(0.1 * i, 1) for i in range(11)]
    plt.yticks(ticks=y_ticks)

    plt.legend(loc='upper left')
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()

    pdf_path = os.path.join(base_dir, f'global_validation_accuracy_alpha_{alpha}.png')
    plt.savefig(pdf_path, format='png')
    
    # ✅ 保存为 PDF 矢量图格式
    pdf_path = os.path.join(base_dir, f'global_validation_accuracy_alpha_{alpha}.pdf')
    plt.savefig(pdf_path, format='pdf')
    plt.close()