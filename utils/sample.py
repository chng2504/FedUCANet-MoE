import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch_npu
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

dotenv.load_dotenv()

DEFAULT_IMAGE_SIZE = 224

CAR_HACKING_IMAGE_DATASET_PATH = "/opt/dpcvol/datasets/1681502372209677867/train"
CIC_IOV2024_IMAGE_DATASET_PATH = "/opt/dpcvol/datasets/6341789061772014206/ciciov2024"

from typing import List

def merge_client_datasets(client_datasets: List["ClientDataset"]) -> "ClientDataset":
    """
    合并多个客户端的 ClientDataset，返回一个新的 ClientDataset，包含所有样本（去重）
    """
    merged_indices = []
    for ds in client_datasets:
        merged_indices.extend(ds.client_indices)
    if not client_datasets:
        raise ValueError("client_datasets 不能为空")
    original_dataset = client_datasets[0].original_dataset
    merged_indices = list(set(merged_indices))  # 去重
    return ClientDataset(original_dataset, merged_indices)


class ImageDataset(datasets.ImageFolder):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        # 如果没有指定transform，使用默认的数据预处理
        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
                        interpolation=transforms.InterpolationMode.LANCZOS,
                    ),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    #                   std=[0.229, 0.224, 0.225])
                ]
            )

        super().__init__(root, transform)

        # 保存类别映射
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        print(f"{root}-找到以下类别: {self.classes}")
        print(f"{root}-每个类别的样本数: {self._get_samples_per_class()}")

    def _get_samples_per_class(self) -> dict:
        """统计每个类别的样本数量"""
        class_counts = {}
        for path, idx in self.samples:
            class_name = self.idx_to_class[idx]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image, label = super().__getitem__(index)
        return image, label

    def __len__(self) -> int:
        return len(self.samples)


class FLDataPartitioner:
    def __init__(self, dataset: datasets.ImageFolder, num_clients: int, seed: int =42):
        self.dataset = dataset
        self.num_clients = num_clients
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 获取类别到索引的映射
        self.class_to_idx = dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # 构建类别样本索引字典
        self.class_indices = defaultdict(list)
        for idx, (path, class_idx) in enumerate(dataset.samples):
            class_name = self.idx_to_class[class_idx]
            self.class_indices[class_name].append(idx)

    def _get_class_distribution(self, indices: List[int]) -> Dict[str, float]:
        """计算给定索引集合的类别分布"""
        class_counts = defaultdict(int)
        for idx in indices:
            _, class_idx = self.dataset.samples[idx]
            class_name = self.idx_to_class[class_idx]
            class_counts[class_name] += 1
        return {k: v for k, v in class_counts.items()}

    def iid_split(self) -> List[List[int]]:
        """IID划分：均匀随机分布所有样本"""
        # 合并所有样本索引
        all_indices = np.arange(len(self.dataset))
        np.random.shuffle(all_indices)

        # 均匀划分
        split_indices = np.array_split(all_indices, self.num_clients)
        return [indices.tolist() for indices in split_indices]

    def non_iid_split(
        self,
        alpha: float = 1.0,
        min_samples: int = 20,
    ) -> List[List[int]]:
        """
        Non-IID划分策略改进版：
        - 正常样本(benign)保持IID均匀分配
        - 异常样本按类别分组后分别进行狄利克雷分配
        - 低alpha值时，异常样本在数量和类别上都更不均衡
        """
        # 分离正常类
        normal_indices = self.class_indices["benign"].copy()

        # 按类别分组异常样本
        anomaly_classes = {}
        for cls in self.class_indices:
            if cls != "benign":
                anomaly_classes[cls] = self.class_indices[cls].copy()

        # 确保最小样本数要求
        total_samples = len(normal_indices) + sum(
            len(v) for v in anomaly_classes.values()
        )
        required_samples = self.num_clients * min_samples
        if total_samples < required_samples:
            min_samples = max(1, total_samples // self.num_clients)

        # 正常样本IID分配
        np.random.shuffle(normal_indices)
        normal_dist = np.array_split(normal_indices, self.num_clients)
        normal_dist = [indices.tolist() for indices in normal_dist]

        # 异常样本按类别分组进行狄利克雷分配
        anomaly_dist = [[] for _ in range(self.num_clients)]
        for cls, indices in anomaly_classes.items():
            if len(indices) > 0:
                # 对每个异常类别单独进行狄利克雷分配
                cls_dist = self._dirichlet_split_no_repeat(indices, alpha=alpha)
                for client_idx in range(self.num_clients):
                    anomaly_dist[client_idx].extend(cls_dist[client_idx])

        # 合并分配结果
        client_indices = []
        unused_indices = normal_indices.copy() + [
            idx for cls in anomaly_classes for idx in anomaly_classes[cls]
        ]

        for i, (n_idx, a_idx) in enumerate(zip(normal_dist, anomaly_dist)):
            combined = n_idx + a_idx

            # 确保最小样本数
            if len(combined) < min_samples and len(unused_indices) > 0:
                needed = min_samples - len(combined)
                extra = min(needed, len(unused_indices))
                np.random.shuffle(unused_indices)
                combined.extend(unused_indices[:extra])
                unused_indices = unused_indices[extra:]

            np.random.shuffle(combined)
            client_indices.append(combined)
            

        return client_indices

    def _dirichlet_split_no_repeat(
        self, indices: List[int], alpha: float
    ) -> List[List[int]]:
        """改进的狄利克雷分配，确保低alpha时更不均衡"""
        if not indices:
            return [[] for _ in range(self.num_clients)]

        indices = indices.copy()
        np.random.shuffle(indices)

        # 生成更极端的分配比例（alpha越小越不均衡）
        proportions = np.random.dirichlet(np.repeat(alpha, self.num_clients))

        # 对比例进行指数变换增强不均衡性
        if alpha < 1.0:
            proportions = np.power(proportions, 1 / alpha)
            proportions /= proportions.sum()

        # 分配样本
        allocation = (proportions * len(indices)).astype(int)
        diff = len(indices) - allocation.sum()

        # 调整分配数量
        if diff > 0:
            allocation[np.argmax(proportions)] += diff
        elif diff < 0:
            allocation[np.argmin(proportions)] += diff

        # 实际分配
        result = []
        start = 0
        for size in allocation:
            end = start + size
            result.append(
                indices[start:end] if end <= len(indices) else indices[start:]
            )
            start = end

        return result

    def print_distribution(
        self, client_indices: List[List[int]], enable_benign: bool = False
    ):
        """可视化各客户端的数据分布"""
        for client_id, indices in enumerate(client_indices):
            dist = self._get_class_distribution(indices)
            print(f"Client {client_id} ({len(indices)} samples):")
            for cls, count in sorted(dist.items(), key=lambda x: x[1], reverse=True):
                # skip benign
                if cls == "benign":
                    if not enable_benign:
                        continue
                print(f"  {cls}: {count}")
            print("-" * 50)

    def visualize_distribution_plot(
        self,
        client_indices: List[List[int]],
        figsize=(16, 8),
        title: str = "Client Data Distribution",
    ) -> plt.Figure:
        """可视化各客户端的数据分布（Seaborn样式的堆叠条形图）"""
        # 获取所有类别列表
        all_classes = [cls for cls in list(self.class_to_idx.keys()) if cls != "benign"]

        # 收集每个客户端的数据分布
        client_data = []
        for client_id, indices in enumerate(client_indices):
            class_counts = defaultdict(int)
            for idx in indices:
                _, class_idx = self.dataset.samples[idx]
                class_name = self.idx_to_class[class_idx]
                if class_name == "benign":
                    continue
                class_counts[class_name] += 1

            # 填充所有类别的计数（包括0值的类别）
            counts = {cls: class_counts.get(cls, 0) for cls in all_classes}
            counts["client"] = f"Client {client_id}"
            client_data.append(counts)
            
        # custom_colors = {
        #     "dos": "#66c2a5",           # 红色
        #     "gas": "#fc8d62",           # 蓝色
        #     "rpm": "#8da0cb",           # 绿色
        #     "speed": "#a6d854",         # 黄色
        #     "steering_wheel": "#ffd92f"  # 粉色
        #  }
        
        custom_colors = {
            "dos": "#66c2a5",           # 红色
            "fuzzy": "#fc8d62",           # 蓝色
             "gear": "#8da0cb",           # 绿色
             "rpm": "#ffd92f",         # 黄色
         }

        # 转换为DataFrame (宽格式数据)
        df = pd.DataFrame(client_data).set_index("client")
        
        
        df = df[[cls for cls in custom_colors.keys()]]

        
        # 创建图形
        plt.figure(figsize=figsize)

        # 使用Matplotlib创建堆叠条形图，但带有Seaborn样式
        ax = df.plot.bar(
            stacked=True,
            color=[custom_colors[cls] for cls in df.columns],
            width=0.8,
            edgecolor="white",
            linewidth=0.5,
        )

        # 设置图形参数
        plt.title(title, fontsize=12, fontweight="semibold")
        plt.xlabel("", fontsize=12, labelpad=10)  # 移除x轴标签
        plt.ylabel("Count", fontsize=11, fontweight="semibold")
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)

        # 只保留横向网格线，移除竖向网格线
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.grid(axis="x", visible=False)

        # 调整图例
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(
            fontsize=8,
            title_fontsize=10,
            frameon=True,
            framealpha=0.9,
            edgecolor="lightgray",
        )

        # 调整布局
        plt.tight_layout()
        return plt,df


class ClientDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, client_indices):
        self.original_dataset = original_dataset
        self.client_indices = client_indices
        self.samples = [original_dataset.samples[i] for i in client_indices]
        self.targets = [original_dataset.targets[i] for i in client_indices]

    def __getitem__(self, index):
        global_idx = self.client_indices[index]
        return self.original_dataset[global_idx]

    def __len__(self):
        return len(self.client_indices)

    def get_paths_labels(self):
        """获取当前客户端所有样本的路径和标签"""
        return [
            (self.original_dataset.samples[i][0], self.original_dataset.targets[i])
            for i in self.client_indices
        ]

    def print_distribution(self):
        """显示当前客户端数据集中各个类别的分布情况"""
        # 收集每个类别的样本数量
        class_counts = defaultdict(int)
        for target in self.targets:
            class_name = self.original_dataset.idx_to_class[target]
            class_counts[class_name] += 1

        # 计算总样本数
        total_samples = len(self.targets)

        # 打印分布信息
        print(f"数据集样本总数: {total_samples}")
        print("各类别样本分布:")
        for cls, count in sorted(
            class_counts.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / total_samples) * 100
            print(f"  {cls}: {count} 样本 ({percentage:.2f}%)")

    def split(self, ratio: float) -> Tuple["ClientDataset", "ClientDataset"]:
        # 按类别收集索引
        class_indices = defaultdict(list)
        for idx, target in enumerate(self.targets):
            class_indices[target].append(self.client_indices[idx])

        first_indices = []
        second_indices = []

        # 对每个类别进行分割
        for class_idx, indices in class_indices.items():
            n_samples = len(indices)
            n_first = int(n_samples * ratio)

            # 随机打乱当前类别的索引
            shuffled_indices = np.random.permutation(indices)

            # 分配到两个子集
            first_indices.extend(shuffled_indices[:n_first])
            second_indices.extend(shuffled_indices[n_first:])

        # 创建新的 ClientDataset 实例
        return (
            ClientDataset(self.original_dataset, first_indices),
            ClientDataset(self.original_dataset, second_indices),
        )


if __name__ == "__main__":
    train_ds = ImageDataset(os.path.join(CIC_IOV2024_IMAGE_DATASET_PATH, "train"))
    client_num = 20

    partitioner = FLDataPartitioner(train_ds, client_num)
    client_indices = partitioner.non_iid_split(alpha=1.0)
    
    # 确保目录存在
    os.makedirs(path, exist_ok=True)

    # 构造文件名
    filename = f"client_indices_alpha{alpha}_{dataset}.csv"
    filepath = os.path.join(path, filename)

    # 转换为 DataFrame
    df = pd.DataFrame.from_dict(
        {f"Client_{i}": indices for i, indices in enumerate(client_indices)},
        orient='index'
    ).transpose()

    # 保存为 CSV
    df.to_csv(filepath, index=False)

    print(f"已保存到: {filepath}")

    client_datasets = [
        ClientDataset(train_ds, client_indices[i]) for i in range(client_num)
    ]

    client_dataloaders = [
        DataLoader(client_datasets[i], batch_size=128, shuffle=True)
        for i in range(client_num)
    ]
