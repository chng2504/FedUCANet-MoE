import os
import pandas as pd

# 设置你的根目录路径
root_folder = '.'  # ←←← 修改为你的主目录路径

# 递归遍历所有文件夹和文件
for root, dirs, files in os.walk(root_folder):
    for filename in files:
        if filename.endswith(".csv"):
            file_path = os.path.join(root, filename)
            try:
                df = pd.read_csv(file_path)

                if df.shape[1] >= 2:
                    old_name = df.columns[1]
                    df.rename(columns={old_name: "global/validate-acc"}, inplace=True)
                    df.to_csv(file_path, index=False)
                    print(f"[✓] 已修改: {file_path}")
                else:
                    print(f"[!] 跳过（列数不足）: {file_path}")
            except Exception as e:
                print(f"[×] 错误处理 {file_path}: {e}")
