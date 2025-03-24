import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from typing import Optional, Callable, Tuple, List, Dict
from collections import defaultdict
from matplotlib.cm import get_cmap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_IMAGE_SIZE = 224


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
    def __init__(self, dataset: datasets.ImageFolder, num_clients: int, seed: int = 42):
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
        alpha: float = 0.5,
        benign_ratio: float = 0.8,
        minority_classes: List[str] = ['gas', 'steering_wheel'],
        min_samples: int = 15
    ) -> List[List[int]]:
        """
        Non-IID划分策略：
        - 使用分层Dirichlet分布控制数据偏斜
        - 保证每个客户端包含足够多的正常样本
        - 特殊处理小样本类别
        """
        # 分离正常类和异常类
        normal_indices = self.class_indices['benign']
        anomaly_indices = []
        for cls in self.class_indices:
            if cls != 'benign':
                anomaly_indices.extend(self.class_indices[cls])

        # 划分正常样本（使用更集中的分布）
        normal_dist = self._dirichlet_split(normal_indices, alpha=alpha/2)
      
        # 划分异常样本（使用更均匀的分布）
        anomaly_dist = self._dirichlet_split(anomaly_indices, alpha=alpha*2)

        # 合并并保证最小样本量
        client_indices = []
        for n_idx, a_idx in zip(normal_dist, anomaly_dist):
            combined = n_idx + a_idx
            if len(combined) < min_samples:
                # 补充随机样本
                extra = np.random.choice(
                    np.concatenate([normal_indices, anomaly_indices]),
                    size=min_samples - len(combined),
                    replace=False
                )
                combined.extend(extra)
            np.random.shuffle(combined)
            client_indices.append(combined)

        # 处理小样本类别：确保出现在足够多的客户端
        for minority_cls in minority_classes:
            if minority_cls in self.class_indices:
                cls_indices = self.class_indices[minority_cls]
                target_clients = np.random.choice(
                    self.num_clients, 
                    size=int(self.num_clients*0.3),  # 至少出现在30%的客户端
                    replace=False
                )
                for client_id in target_clients:
                    client_indices[client_id].extend(cls_indices)

        return client_indices

    def _dirichlet_split(self, indices: List[int], alpha: float) -> List[List[int]]:
        """基于Dirichlet分布的样本划分"""
        if not indices:
            return [[] for _ in range(self.num_clients)]

        # 生成分配比例矩阵
        proportions = np.random.dirichlet(np.repeat(alpha, self.num_clients))
        proportions /= np.sum(proportions)

        # 计算分配数量
        num_samples = len(indices)
        allocations = (np.cumsum(proportions) * num_samples).astype(int)[:-1]
        split_indices = np.split(np.random.permutation(indices), allocations)
      
        # 填充结果
        return [list(indices) for indices in split_indices]

    def print_distribution(self, client_indices: List[List[int]], enable_benign: bool = False):
        """可视化各客户端的数据分布"""
        for client_id, indices in enumerate(client_indices):
            dist = self._get_class_distribution(indices)
            print(f"Client {client_id} ({len(indices)} samples):")
            for cls, count in sorted(dist.items(), key=lambda x: x[1], reverse=True):
                # skip benign
                if cls == 'benign':
                    if not enable_benign:
                        continue
                print(f"  {cls}: {count}")
            print("-"*50)
            
    def visualize_distribution_plot(self, 
                                    client_indices: List[List[int]], 
                                    figsize=(16, 8),
                                    title: str = 'Client Data Distribution',
                                    ):
        """可视化各客户端的数据分布（多色条形图版）"""
        # 获取所有类别列表
        all_classes = list(self.class_to_idx.keys())
    
        # 收集每个客户端的数据分布
        client_data = []
        for client_id, indices in enumerate(client_indices):
            class_counts = defaultdict(int)
            for idx in indices:
                # skip benign
                _, class_idx = self.dataset.samples[idx]
                class_name = self.idx_to_class[class_idx]
                if class_name == 'benign':
                    continue
                class_counts[class_name] += 1
        
            # 填充所有类别的计数（包括0值的类别）
            # skip benign
            counts = {cls: class_counts.get(cls, 0) for cls in all_classes if cls != 'benign'}
            counts['client'] = f'Client {client_id}'
            client_data.append(counts)

        # 转换为DataFrame
        df = pd.DataFrame(client_data).set_index('client')
    
        # 创建颜色映射
        colormap = get_cmap('tab20')
        colors = [colormap(i % 20) for i in range(len(all_classes))]
    
        # 创建堆叠条形图
        plt.figure(figsize=figsize)
        ax = df.plot.bar(
            stacked=True,
            color=colors,
            width=0.8,
            edgecolor='white',
            linewidth=0.5
        )

        # 设置图形参数
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('Client ID', fontsize=12, labelpad=10)
        plt.ylabel('Sample Count', fontsize=12, labelpad=10)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
    
        # 调整图例
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(
            reversed(handles),
            reversed(labels),
            title='Classes',
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            fontsize=8,
            title_fontsize=10
        )


        # 调整布局
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        plt.show()


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
        
        
        

    
    