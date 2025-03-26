import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import accelerate
import dotenv
import os
from torch import nn, optim
from models.mvn4 import MVN4TrimNet, GateTrimNet
from torch.utils.data import DataLoader
from utils import sample
from tqdm import tqdm
from loguru import logger
import swanlab
import argparse
import random
from typing import Dict, List
import numpy as np
import torch_npu
import clientor
import aggregator

dotenv.load_dotenv()

CAR_HACKING_IMAGE_DATASET_PATH = os.getenv("CAR_HACKING_IMAGE_DATASET")
CIC_IOV2024_IMAGE_DATASET_PATH = os.getenv("CIC_IOV2024_IMAGE_DATASET")
CUR_DATASET = "ciciov2024"


GLOBAL_ACCELERATOR = accelerate.Accelerator()
device = GLOBAL_ACCELERATOR.device

PROJECT_NAME: str = "FL-MoE"
EXPERIMENT_NAME: str = "2025-03-26"
DESCRIPTION: str = "联邦MoE模型训练"
sw_config = {
    "learning_rate": 5e-4,
    "batch_size": 128,
    "device": GLOBAL_ACCELERATOR.device,
    "num_clients": 10,
    "num_classes": 6 if CUR_DATASET == "ciciov2024" else 5,
    "num_workers": 8,
    "cur_dataset": CUR_DATASET,
    "client_num": 20,
    "client_per_round": 5,
    "client_evaluate": 10,
    "global_rounds": 10,
    "local_rounds": 3,
    "learning_rate": 1e-5,
}


def train_client(
    idx,
    accelerator,
    epochs,
    model,
    train_ds: sample.ClientDataset,
    learning_rate=sw_config.get("learning_rate"),
    enable_swanlab=False,
):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        train_ds,
        batch_size=sw_config.get("batch_size"),
        shuffle=True,
        num_workers=sw_config.get("num_workers"),
    )
    model, optimizer, train_loader, criterion = accelerator.prepare(
        model, optimizer, train_loader, criterion
    )

    model_best = model.state_dict()
    train_acc_best = 0.0
    for epoch in range(epochs):
        total_loss = 0.0
        y_true = []
        y_pred = []
        for idx, (images, labels) in enumerate(
            tqdm(train_loader, desc=f"Client {idx} - Epoch [{epoch + 1}/{epochs}]")
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

        train_loss = total_loss / len(train_loader)
        avg_acc = accuracy_score(y_true, y_pred)
        if enable_swanlab:
            swanlab.log(
                {
                    f"client-fed-train/{idx}-train_loss": train_loss,
                    f"client-fed-train/{idx}-train_acc": avg_acc,
                }
            )
        accelerator.print(
            f"Epoch [{epoch + 1}/{epochs}] - Loss: {train_loss}, acc: {avg_acc}"
        )
        if avg_acc > train_acc_best:
            train_acc_best = avg_acc
            model_best = model.state_dict()
    return model_best, train_acc_best


def train_finetune(
    idx,
    accelerator,
    local_epochs,
    model,
    train_ds: sample.ClientDataset,
    learning_rate=sw_config.get("learning_rate"),
    enable_swanlab=False,
):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        train_ds,
        batch_size=sw_config.get("batch_size"),
        shuffle=True,
        num_workers=sw_config.get("num_workers"),
    )
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
        if enable_swanlab:
            swanlab.log(
                {
                    f"client-finetune-train/{idx}-train_loss": epoch_loss,
                    f"client-finetune-train/{idx}-train_acc": avg_acc,
                }
            )
        accelerator.print(
            f"Local Epoch [{epoch + 1}/{local_epochs}] - Loss: {epoch_loss}, acc: {avg_acc}"
        )
        if avg_acc > train_acc_best:
            train_acc_best = avg_acc
            model_best = model.state_dict()
    return model_best, epoch_loss


def train_mix(
    idx,
    accelerator,
    local_epochs,
    model_global,
    model_local,
    model_gate,
    train_ds: sample.ClientDataset,  # 这里用的所有数据集
    train_gate_only=False,
    enable_swanlab=False,
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
        optimizer = optim.Adam(
            model_gate.parameters(), lr=sw_config.get("learning_rate")
        )
    else:
        optimizer = optim.Adam(
            list(model_local.parameters()) + list(model_gate.parameters()),
            lr=sw_config.get("learning_rate"),
        )

    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        train_ds,
        batch_size=sw_config.get("batch_size"),
        shuffle=True,
        num_workers=sw_config.get("num_workers"),
    )
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
        if enable_swanlab:
            swanlab.log(
                {
                    f"client-mix-train/{idx}-train_loss": epoch_loss,
                    f"client-mix-train/{idx}-train_acc": avg_acc,
                }
            )
        accelerator.print(
            f"Epoch [{epoch + 1}/{local_epochs}] - Loss: {epoch_loss}, acc: {avg_acc}"
        )
        if avg_acc > train_acc_best:
            train_acc_best = avg_acc
            gate_best = model_gate.state_dict()
            local_best = model_local.state_dict()
            global_best = model_global.state_dict()

    return gate_best, local_best, global_best, epoch_loss, train_acc_best


def validate(
    idx, accelerator, model, test_ds: sample.ClientDataset, enable_swanlab=False
):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    y_true = []
    y_pred = []
    total_loss = 0.0
    test_loader = DataLoader(
        test_ds,
        batch_size=sw_config.get("batch_size"),
        shuffle=False,
        num_workers=sw_config.get("num_workers"),
    )
    model, test_loader, criterion = accelerator.prepare(model, test_loader, criterion)
    with torch.no_grad():
        for idx, (images, labels) in enumerate(
            tqdm(test_loader, desc="Validating Single")
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            cur_pred = predicted.cpu().numpy()
            cur_label = labels.cpu().numpy()
            y_true.extend(cur_label)
            y_pred.extend(cur_pred)
    avg_acc = accuracy_score(y_true, y_pred)
    avg_loss = total_loss / len(test_loader)
    if enable_swanlab:
        swanlab.log(
            {
                f"client-validate/{idx}-test_loss": avg_loss,
                f"client-validate/{idx}-test_acc": avg_acc,
            }
        )
    accelerator.print(f"Validating Global - Loss: {avg_loss}, acc: {avg_acc}")
    return avg_acc, avg_loss


def validate_mix(
    idx,
    accelerator,
    model_global,
    model_local,
    model_gate,
    test_ds: sample.ClientDataset,
    enable_swanlab=False,
):
    model_local.eval()
    model_global.eval()
    model_gate.eval()

    criterion = nn.CrossEntropyLoss()
    y_true = []
    y_pred = []
    total_loss = 0.0
    test_loader = DataLoader(
        test_ds,
        batch_size=sw_config.get("batch_size"),
        shuffle=False,
        num_workers=sw_config.get("num_workers"),
    )
    model_local, model_global, model_gate, test_loader, criterion = accelerator.prepare(
        model_local, model_global, model_gate, test_loader, criterion
    )
    with torch.no_grad():
        for idx, (images, labels) in enumerate(
            tqdm(test_loader, desc="Validating Mix")
        ):
            gate_weight = model_gate(images)
            local_prob = model_local(images)
            global_prob = model_global(images)

            log_probs = gate_weight * local_prob + (1 - gate_weight) * global_prob
            loss = criterion(log_probs, labels)
            total_loss += loss.item()

            cur_pred = log_probs.argmax(dim=1).cpu().numpy()
            cur_label = labels.cpu().numpy()
            y_true.extend(cur_label)
            y_pred.extend(cur_pred)

    avg_loss = total_loss / len(test_loader)
    avg_acc = accuracy_score(y_true, y_pred)
    if enable_swanlab:
        swanlab.log(
            {
                f"client-mix-validate/{idx}-test_loss": avg_loss,
                f"client-mix-validate/{idx}-test_acc": avg_acc,
            }
        )
    accelerator.print(f"Validating Mix - Loss: {avg_loss}, acc: {avg_acc}")
    return avg_acc, avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--swanlab", type=bool, default=False)
    parser.add_argument("--dataset", type=str, default="ciciov2024")
    parser.add_argument("--out_ratio", type=float, default=0.0)
    args = parser.parse_args()

    if args.swanlab:
        swanlab.init(
            project_name=PROJECT_NAME,
            experiment_name=EXPERIMENT_NAME,
            description=DESCRIPTION,
            config=sw_config,
        )
    global_train_ds = sample.ImageDataset(f"{CIC_IOV2024_IMAGE_DATASET_PATH}/train")
    global_test_ds = sample.ImageDataset(f"{CIC_IOV2024_IMAGE_DATASET_PATH}/test")
    clients, ratio_list = clientor.prepare_client_datasets(
        train_ds=global_train_ds,
        test_ds=global_test_ds,
        client_num=20,
        split_type=clientor.SplitType.NON_IID,
        global_rate=0.5,
    )

    global_model = MVN4TrimNet(num_classes=sw_config.get("num_classes"))

    for client in clients:
        client.create_models(sw_config.get("num_classes"))

    # 客户端字典：idx -> client
    client_dict: Dict[int, clientor.Client] = {client.idx: client for client in clients}
    fed_global_model_weight = global_model.state_dict()
    for round in range(sw_config.get("global_rounds")):
        print(
            f"========== Global Round [{round + 1} / {sw_config.get('global_rounds')}] Start =========="
        )
        cur_clients: List[clientor.Client] = random.sample(
            clients, sw_config.get("client_per_round")
        )
        # ! 1. 第一次加载全局模型
        for client in cur_clients:
            client.model_global.load_state_dict(fed_global_model_weight)
        cur_client_weights: List[Dict[str, torch.Tensor]] = []
        cur_client_idxeds = [client.idx for client in cur_clients]

        cur_ratio_list = ratio_list[cur_client_idxeds]
        cur_ratio_list = cur_ratio_list / np.sum(cur_ratio_list)

        logger.info(f"Current Clients: {cur_client_idxeds}, Ratio: {cur_ratio_list}")
        for client in cur_clients:
            if device == "npu":
                torch_npu.npu.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
            else:
                pass
            logger.info(f"Training Client {client.idx}...")
            cur_weight, _ = train_client(
                client.idx,
                GLOBAL_ACCELERATOR,
                sw_config.get("local_rounds"),
                client.model_global,
                client.global_ds,
                sw_config.get("learning_rate"),
                args.swanlab,
            )
            cur_client_weights.append(cur_weight)

        #! 2. 聚合并第二次加载全局模型
        fed_global_model_weight = aggregator.FedAvg(cur_client_weights, cur_ratio_list)
        global_model.load_state_dict(fed_global_model_weight)
        for client in cur_clients:
            client.model_global.load_state_dict(fed_global_model_weight)


if __name__ == "__main__":
    main()
