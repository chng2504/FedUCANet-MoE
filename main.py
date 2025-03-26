import accelerate
import dotenv
import os
from loguru import logger
import swanlab
import clientor
from utils import sample
from typing import List, Dict, Any
from models.mvn4 import MVN4TrimNet, GateTrimNet
import trainer
import aggregator
import random
import torch_npu
from tqdm import tqdm
from loguru import logger
import torch
import numpy as np

dotenv.load_dotenv()
GLOBAL_ACCELERATOR = accelerate.Accelerator()

device = GLOBAL_ACCELERATOR.device

CAR_HACKING_IMAGE_DATASET_PATH = os.getenv("CAR_HACKING_IMAGE_DATASET")
CIC_IOV2024_IMAGE_DATASET_PATH = os.getenv("CIC_IOV2024_IMAGE_DATASET")

CUR_DATASET = "ciciov2024"
CONFIG = {
    "num_classes": 6 if CUR_DATASET == "ciciov2024" else 5,
    "client_num": 20,
    "client_per_round": 5,
    "client_evaluate": 10,
    "global_rounds": 10,
    "local_rounds": 3,
    "learning_rate": 1e-5,
}


def main():
    global_train_ds = sample.ImageDataset(f"{CIC_IOV2024_IMAGE_DATASET_PATH}/train")
    global_test_ds = sample.ImageDataset(f"{CIC_IOV2024_IMAGE_DATASET_PATH}/test")
    clients, ratio_list = clientor.prepare_client_datasets(
        train_ds=global_train_ds,
        test_ds=global_test_ds,
        client_num=20,
        split_type=clientor.SplitType.NON_IID,
        global_rate=0.5,
    )

    global_model = MVN4TrimNet(num_classes=CONFIG["num_classes"])

    for client in clients:
        client.model_local = MVN4TrimNet(num_classes=CONFIG["num_classes"])
        client.model_global = MVN4TrimNet(num_classes=CONFIG["num_classes"])
        client.model_gate = GateTrimNet()

    # 客户端字典：idx -> client
    client_dict: Dict[int, clientor.Client] = {client.idx: client for client in clients}
    fed_global_model_weight = global_model.state_dict()
    for round in range(CONFIG["global_rounds"]):
        print(
            f"========== Global Round [{round + 1} / {CONFIG['global_rounds']}] Start =========="
        )
        cur_clients: List[clientor.Client] = random.sample(
            clients, CONFIG["client_per_round"]
        )
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
            cur_weight, _ = trainer.train_one_epoch(
                GLOBAL_ACCELERATOR,
                CONFIG["local_rounds"],
                client.model_global,
                client.global_ds,
                CONFIG["learning_rate"],
            )
            cur_client_weights.append(cur_weight)

        fed_global_model_weight = aggregator.FedAvg(cur_client_weights, cur_ratio_list)
        global_model.load_state_dict(fed_global_model_weight)


if __name__ == "__main__":
    main()
