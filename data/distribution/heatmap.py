import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap

# 读取并转置数据
df = pd.read_csv('csv_distributions\client_distribution_alpha10_0.csv', index_col=0).T
data = df.values
data = df.div(df.sum(axis=0), axis=1).values
clients = df.columns.tolist()
categories = df.index.tolist()

# ✅ 去掉 client 前缀，只保留编号
clients = [c.replace("Client ", "") for c in clients]

# 你指定的三个颜色（可以用名字或 RGB 值）
colors = ['#FDFEDC','#73C674','#04462A']  # 浅黄 → 橙 → 深红
# 创建 colormap，256表示渐变层级越细腻
custom_cmap = LinearSegmentedColormap.from_list("my_colormap", colors, N=256)


# 每格宽高设置
cell_w, cell_h = 0.3, 0.8
rows, cols = data.shape
figsize = (cols * cell_w, rows * cell_h)
fig, ax = plt.subplots(figsize=figsize)

# ✅ 设置坐标轴范围，彻底打破默认格子比例
extent = [0, cols, rows, 0]  # 左右上下（注意 y 是反的）
im = ax.imshow(data, cmap=custom_cmap, aspect='auto', extent=extent)

# 设置刻度
ax.set_xticks(np.arange(cols) + 0.5)
ax.set_yticks(np.arange(rows) + 0.5)
ax.set_xticklabels(clients)

ax.set_yticklabels(categories)
plt.setp(ax.get_xticklabels(), rotation=0, ha='right')

# ✅ 添加 colorbar，保持等高
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax)
ax.set_xlabel("Client")

# ✅ 添加网格线
ax.set_xticks(np.arange(cols + 1), minor=True)
ax.set_yticks(np.arange(rows + 1), minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)

# plt.title("",loc="center")
fig.suptitle("Local/Distribution", x=0.5, y=0.95, ha='center')
plt.tight_layout()
plt.savefig("alpha=10")
plt.show()
