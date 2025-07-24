import pandas as pd
import matplotlib.pyplot as plt

# 加载CSV文件
df = pd.read_csv("accuracy.csv")  # 替换为你的CSV文件路径
df=df.head(500)

# 提取 step 列
steps = df["step"]


# 原始列名与目标顺序对应的映射（必须和实际列数一一对应）
original_cols = df.columns[1:]  # 排除 step 列
custom_labels = [
    "EMO",
    "ConvNext",
    "MobileViT",
    "MobileNetV3",
    "ResNet18",
    "VGG-16",
    "UCANet"
]

# 你希望的显示顺序（根据 custom_labels 中的内容重排）
desired_order = [
    "VGG-16",
    "ResNet18",
    "MobileNetV3",
    "MobileViT",
    "ConvNext",
    "EMO",
    "UCANet"
]

# 建立 label 到原始列名的映射
label_to_col = dict(zip(custom_labels, original_cols))

# 初始化画布
plt.figure(figsize=(12, 6))

# 按照你设定的顺序绘图
for label in desired_order:
    col = label_to_col[label]
    plt.plot(steps, df[col], label=label)

# 图像美化
plt.title("Train/Accuracy", fontsize=14)
plt.xlabel("Step", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("ucanet_accuracy")

# 显示图像
plt.show()
