import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import accelerate
import dotenv
import os
from torch import nn, optim
from models.mvn4 import MVN4TrimNet, GateTrimNet
from torch.utils.data import DataLoader
from utils.sample import ImageDataset
from tqdm import tqdm
from loguru import logger

dotenv.load_dotenv()
import numpy as np


CAR_HACKING_IMAGE_DATASET_PATH = os.getenv("CAR_HACKING_IMAGE_DATASET")
CIC_IOV2024_IMAGE_DATASET_PATH = os.getenv("CIC_IOV2024_IMAGE_DATASET")

CUR_DATASET = "ciciov2024"
GLOBAL_ACCELERATOR = accelerate.Accelerator()
CONFIG = {
    "learning_rate": 1e-5,
    "batch_size": 128,
    "device": GLOBAL_ACCELERATOR.device,
    "num_clients": 10,
    "num_class": 5 if CUR_DATASET == "carhacking" else 6,
    "epochs": 10,
    "num_workers": 8,
}


# ! TODO: 这是全局的数据集，后面需要按照采样策略分成多个 client 的数据集
def prepare_dataset(
    dataset_name: str = "carhacking", batch_size: int = 128, num_workers: int = 8
):
    parent_path: str = ""
    if dataset_name == "carhacking":
        parent_path = CAR_HACKING_IMAGE_DATASET_PATH
    elif dataset_name == "ciciov2024":
        parent_path = CIC_IOV2024_IMAGE_DATASET_PATH
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_ds = ImageDataset(root=os.path.join(parent_path, "train"))
    test_ds = ImageDataset(root=os.path.join(parent_path, "test"))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def train_one_epoch(accelerator, epoch, model, train_loader):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    model, optimizer, train_loader, criterion = accelerator.prepare(
        model, optimizer, train_loader, criterion
    )

    y_true = []
    y_pred = []
    total_loss = 0.0
    last_loss = 0.0
    for idx, (images, labels) in enumerate(
        tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{CONFIG['epochs']}]")
    ):
        optimizer.zero_grad()
        # outputs = nn.Softmax(dim=-1)(model(images))
        outputs = model(images)
        loss = criterion(outputs, labels)

        # loss.backward()
        accelerator.backward(loss)

        optimizer.step()
        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        cur_pred = predicted.cpu().numpy()
        cur_label = labels.cpu().numpy()
        y_true.extend(cur_label)
        y_pred.extend(cur_pred)
        last_loss = loss.item()

    train_loss = total_loss / len(train_loader)
    avg_acc = accuracy_score(y_true, y_pred)

    accelerator.print(
        f"Epoch [{epoch + 1}/{CONFIG['epochs']}] - Loss: {train_loss}, acc: {avg_acc}"
    )
    return model.state_dict(), last_loss


def train_finetune(accelerator, local_epochs, model, train_loader):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    model, optimizer, train_loader, criterion = accelerator.prepare(
        model, optimizer, train_loader, criterion
    )
    y_true = []
    y_pred = []
    total_loss = 0.0
    last_loss = 0.0
    model_best = model.state_dict()
    train_acc_best = 0.0
    for epoch in range(local_epochs):
        for idx, (images, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{local_epochs}]")
        ):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            cur_pred = predicted.cpu().numpy()
            cur_label = labels.cpu().numpy()
            y_true.extend(cur_label)
            y_pred.extend(cur_pred)
        epoch_loss = total_loss / len(train_loader)
        avg_acc = accuracy_score(y_true, y_pred)
        accelerator.print(
            f"Local Epoch [{epoch + 1}/{local_epochs}] - Loss: {epoch_loss}, acc: {avg_acc}"
        )
        if avg_acc > train_acc_best:
            train_acc_best = avg_acc
            model_best = model.state_dict()
    return model_best, epoch_loss


def train_mix(
    accelerator,
    local_epochs,
    model_global,
    model_local,
    model_gate,
    train_loader,
    train_gate_only = False
):
    model_global.train()
    model_local.train()
    model_gate.train()
    
    gate_best = model_gate.state_dict()
    local_best = model_local.state_dict()
    global_best = model_global.state_dict()

    y_true = []
    y_pred = []
    train_acc_best = 0.0
    if train_gate_only:
        optimizer = optim.Adam(model_gate.parameters(), lr=CONFIG["learning_rate"])
    else:
        optimizer = optim.Adam(
            list(model_local.parameters()) + list(model_gate.parameters()),
            lr=CONFIG["learning_rate"],
        )

    criterion = nn.CrossEntropyLoss()
    model_global, model_local, model_gate, optimizer, train_loader, criterion = (
        accelerator.prepare(
            model_global, model_local, model_gate, optimizer, train_loader, criterion
        )
    )
    for epoch in range(local_epochs):
        total_loss = 0.0
        for idx, (images, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{local_epochs}]")
        ):
            model_global.zero_grad()
            model_local.zero_grad()
            model_gate.zero_grad()
            
            gate_weight = model_gate(images)
            local_prob = model_local(images)
            global_prob = model_global(images)
            
            log_probs = gate_weight * local_prob + (1 - gate_weight) * global_prob
            loss = criterion(log_probs, labels)
            
            accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
            
            cur_pred = log_probs.argmax(dim=1).cpu().numpy()
            cur_label = labels.cpu().numpy()
            y_true.extend(cur_label)
            y_pred.extend(cur_pred)
            
        epoch_loss = total_loss / len(train_loader)
        avg_acc = accuracy_score(y_true, y_pred)
        accelerator.print(f"Epoch [{epoch + 1}/{local_epochs}] - Loss: {epoch_loss}, acc: {avg_acc}")
        if avg_acc > train_acc_best:
            train_acc_best = avg_acc
            gate_best = model_gate.state_dict()
            local_best = model_local.state_dict()
            global_best = model_global.state_dict()
            
    return gate_best, local_best, global_best, epoch_loss, train_acc_best
            
            
            
    
    


if __name__ == "__main__":
    logger.debug("Start training")

    accelerator = GLOBAL_ACCELERATOR
    print(accelerator.device)

    train_loader, test_loader = prepare_dataset(
        dataset_name=CUR_DATASET,
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )
    model_global = MVN4TrimNet(num_classes=CONFIG["num_class"])
    model_local = MVN4TrimNet(num_classes=CONFIG["num_class"])
    model_gate = GateTrimNet()

    # model._kaiming_init()

    # param = None
    # for epoch in range(CONFIG["epochs"]):
    #     param, last_loss = train_one_epoch(accelerator, epoch, model, train_loader, optimizer, criterion)
    #     assert False

    gate_best, local_best, global_best, epoch_loss, train_acc_best = train_mix(
        accelerator, 3, model_global, model_local, model_gate, train_loader, False
    )
