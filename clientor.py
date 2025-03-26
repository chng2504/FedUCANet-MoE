from torch.utils.data import DataLoader
from torch import nn
from utils import sample
import os
import dotenv
from utils import sample
from typing import List, Tuple
from enum import Enum
import numpy as np
from models.mvn4 import MVN4TrimNet, GateTrimNet

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
    train_ds: sample.ImageDataset,
    test_ds: sample.ImageDataset,
    client_num: int = 20,
    global_rate: float = 0.5,
    alpha: float = 1.0,  # Dirchlet 采样值
    split_type: SplitType = SplitType.NON_IID,
) -> Tuple[List[Client], np.ndarray]:
    train_partitioner = sample.FLDataPartitioner(train_ds, client_num)
    if split_type == SplitType.IID:
        client_train_indices = train_partitioner.iid_split()
    else:
        client_train_indices = train_partitioner.non_iid_split(alpha=alpha)
    client_train_datasets = [
        sample.ClientDataset(train_ds, client_train_indices[i])
        for i in range(client_num)
    ]

    test_partitioner = sample.FLDataPartitioner(test_ds, client_num)
    if split_type == SplitType.IID:
        client_test_indices = test_partitioner.iid_split()
    else:
        client_test_indices = test_partitioner.non_iid_split(alpha=alpha)
    client_test_datasets: List[sample.ClientDataset] = [
        sample.ClientDataset(test_ds, client_test_indices[i]) for i in range(client_num)
    ]

    clients: List[Client] = []
    ratio_list: np.ndarray = np.zeros(client_num)
    for i in range(client_num):
        client = Client(
            i,
            client_train_datasets[i],
            client_test_datasets[i],
            global_rate=global_rate,
        )
        clients.append(client)
        ratio_list[i] = len(client.global_ds) / len(client.train_ds)

    return clients, ratio_list
