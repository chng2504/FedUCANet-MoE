from torch.utils.data import DataLoader
from torch import nn
from utils import sample
import os
import dotenv

dotenv.load_dotenv()



CAR_HACKING_IMAGE_DATASET_PATH = os.getenv("CAR_HACKING_IMAGE_DATASET")
CIC_IOV2024_IMAGE_DATASET_PATH = os.getenv("CIC_IOV2024_IMAGE_DATASET")


class Client:
    def __init__(
        self,
        idx,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model_local: nn.Module,
        model_global: nn.Module,
        model_gate: nn.Module,
    ):  
        self.idx = idx
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_local = model_local
        self.model_global = model_global
        self.model_gate = model_gate
        
    def print_info(self):
        print(f"Client {self.idx} has {len(self.train_loader.dataset)} train samples and {len(self.test_loader.dataset)} test samples")
        



if __name__ == "__main__":
    train_ds = sample.ImageDataset(os.path.join(CIC_IOV2024_IMAGE_DATASET_PATH, "train"))
    test_ds = sample.ImageDataset(os.path.join(CIC_IOV2024_IMAGE_DATASET_PATH, "test"))
    client_num = 20

    train_partitioner = sample.FLDataPartitioner(train_ds, client_num)
    client_train_indices = train_partitioner.non_iid_split(alpha=1.0)
    client_train_datasets = [sample.ClientDataset(train_ds, client_train_indices[i]) for i in range(client_num)]

    client_train_dataloaders = [
        DataLoader(client_train_datasets[i], batch_size=128, shuffle=True)
        for i in range(client_num)
    ]

    test_partitioner = sample.FLDataPartitioner(test_ds, client_num)
    client_test_indices = test_partitioner.non_iid_split(alpha=1.0)
    client_test_datasets = [sample.ClientDataset(test_ds, client_test_indices[i]) for i in range(client_num)]

    client_test_dataloaders = [
        DataLoader(client_test_datasets[i], batch_size=128, shuffle=True)
        for i in range(client_num)
    ]

    for i in range(client_num):
        client = Client(i, client_train_dataloaders[i], client_test_dataloaders[i], None, None, None)
        client.print_info()
