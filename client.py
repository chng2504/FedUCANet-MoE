from torch.utils.data import DataLoader
from torch import nn
from utils import sample
import os
import dotenv
from utils import sample
from typing import List
dotenv.load_dotenv()



CAR_HACKING_IMAGE_DATASET_PATH = os.getenv("CAR_HACKING_IMAGE_DATASET")
CIC_IOV2024_IMAGE_DATASET_PATH = os.getenv("CIC_IOV2024_IMAGE_DATASET")


class Client:
    def __init__(
        self,
        idx,
        train_ds: sample.ClientDataset,
        test_ds: sample.ClientDataset,
        model_local: nn.Module = None,
        model_global: nn.Module = None,
        model_gate: nn.Module = None,
        global_rate: float = 0.5
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
        print(f"Client {self.idx} has {len(self.train_ds)} train samples and {len(self.test_ds)} test samples")
        print(f"Global dataset size: {len(self.global_ds)}")
        print(f"Local dataset size: {len(self.local_ds)}")


if __name__ == "__main__":
    train_ds = sample.ImageDataset(os.path.join(CIC_IOV2024_IMAGE_DATASET_PATH, "train"))
    test_ds = sample.ImageDataset(os.path.join(CIC_IOV2024_IMAGE_DATASET_PATH, "test"))
    client_num = 20

    train_partitioner = sample.FLDataPartitioner(train_ds, client_num)
    client_train_indices = train_partitioner.non_iid_split(alpha=1.0)
    client_train_datasets = [sample.ClientDataset(train_ds, client_train_indices[i]) for i in range(client_num)]


    test_partitioner = sample.FLDataPartitioner(test_ds, client_num)
    client_test_indices = test_partitioner.non_iid_split(alpha=1.0)
    client_test_datasets: List[sample.ClientDataset] = [sample.ClientDataset(test_ds, client_test_indices[i]) for i in range(client_num)]

    clients: List[Client] = []

    for i in range(client_num):
        client = Client(i, client_train_datasets[i], client_test_datasets[i])
        clients.append(client)
        # client.print_info()
        # print("-" * 50)

        
    

