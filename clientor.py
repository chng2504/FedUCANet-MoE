from enum import Enum
from typing import List, Tuple

import dotenv
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from models.mvn4 import GateTrimNet, MVN4TrimNet
from utils import sample

import os
import csv

dotenv.load_dotenv()


class SplitType(Enum):
    IID: str = "iid"
    NON_IID: str = "non_iid"


class Client:
    def __init__(
        self,
        idx,
        train_ds: sample.ClientDataset,
        test_ds: sample.ClientDataset,
        model_local: nn.Module = None,
        model_global: nn.Module = None,
        model_gate: nn.Module = None,
        global_rate: float = 0.5,
    ):
        self.idx = idx
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.model_local = model_local
        self.model_global = model_global
        self.model_gate = model_gate
        self.global_rate = global_rate
        self.global_ds, self.local_ds = self._split_global_local()

    def _split_global_local(self):
        global_ds, local_ds = self.train_ds.split(self.global_rate)
        return global_ds, local_ds

    def print_info(self):
        print(
            f"Client {self.idx} has {len(self.train_ds)} train samples and {len(self.test_ds)} test samples"
        )
        print(f"Global dataset size: {len(self.global_ds)}")
        print(f"Local dataset size: {len(self.local_ds)}")

    def create_models(self, num_classes: int):
        self.model_local = MVN4TrimNet(num_classes=num_classes)
        self.model_global = MVN4TrimNet(num_classes=num_classes)
        self.model_gate = GateTrimNet()


def prepare_client_datasets(
    full_ds: sample.ImageDataset,
    client_num: int = 20,
    test_rate: float = 0.2,  # 每个客户端 20% 数据做测试集
    global_rate: float = 0.5,
    alpha: float = 1.0,
    split_type: SplitType = SplitType.NON_IID,
) -> Tuple[List[Client], np.ndarray, plt.Figure, sample.ClientDataset]:
    """
    准备联邦学习的客户端数据集
    返回：
        clients: 客户端对象列表
        ratio_list: 每个客户端在全体测试集中的比例（用于准确率加权平均）
        dist_fig: 数据分布可视化图
        global_test_ds: 所有客户端测试集汇总而成的全局测试集
    """

    # 1. 使用 Dirichlet 或 IID 分配原始数据索引
    partitioner = sample.FLDataPartitioner(full_ds, client_num)
    if split_type == SplitType.IID:
        client_indices = partitioner.iid_split()
    else:
        client_indices = partitioner.non_iid_split(alpha=alpha)

    print("Client Raw Distribution:")
    partitioner.print_distribution(client_indices)

    # 2. 可视化分布
    plt.figure(1)
    dist_plt, dist_df = partitioner.visualize_distribution_plot(
        client_indices, title="Client Raw Data Distribution"
    )
    output_dir = os.path.join(os.getcwd(), 'distribution')
    os.makedirs(output_dir, exist_ok=True)
    dist_df.to_csv(
        os.path.join(output_dir, f'client_distribution_alpha{str(alpha).replace(".", "_")}.csv')
    )
    dist_fig = dist_plt.gcf()

    # 3. 划分每个客户端的 train/test 数据
    clients = []
    test_sizes = []        # 用于 ratio_list 的计算
    test_indices_all = []  # 用于 global_test_ds 构建

    for i in range(client_num):
        indices = client_indices[i]
        np.random.shuffle(indices)

        num_total = len(indices)
        num_test = int(num_total * test_rate)
        test_indices = indices[:num_test]
        train_indices = indices[num_test:]

        test_indices_all.extend(test_indices)
        test_sizes.append(len(test_indices))

        train_ds = sample.ClientDataset(full_ds, train_indices)
        test_ds = sample.ClientDataset(full_ds, test_indices)

        client = Client(
            idx=i,
            train_ds=train_ds,
            test_ds=test_ds,
            global_rate=global_rate
        )
        clients.append(client)

    # 4. 计算测试集占比（ratio_list 用于加权平均）
    total_test_samples = sum(test_sizes)
    ratio_list = np.array(test_sizes) / total_test_samples

    # 5. 构造全局测试集
    global_test_ds = sample.ClientDataset(full_ds, test_indices_all)

    return clients, ratio_list, dist_fig, global_test_ds
