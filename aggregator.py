from typing import List, Dict
import torch
from models.mvn4 import MVN4TrimNet
from loguru import logger
import numpy as np


def FedAvg(
    weights: List[Dict[str, torch.Tensor]], alpha: np.ndarray
) -> Dict[str, torch.Tensor]:
    """
    Federated Averaging
    weights: List[Dict[str, torch.Tensor]]: list of model weights, representing the weights of each client
    alpha: np.ndarray: list of alpha, representing the importance of each client
    """
    assert len(weights) == alpha.shape[0]

    avg_weights = {}
    for key in weights[0].keys():
        avg_weights[key] = torch.sum(
            torch.stack([alpha[i] * weights[i][key] for i in range(len(weights))]),
            dim=0,
        )

    return avg_weights


if __name__ == "__main__":
    model = MVN4TrimNet(num_classes=6)
    client_num = 20
    logger.info(f"Client number: {client_num}")

    model_list = [model.state_dict() for _ in range(client_num)]
    alpha = np.random.uniform(0, 1, client_num)
    alpha = alpha / alpha.sum()
    print(alpha)
    print(alpha.sum())

    avg_weights = FedAvg(model_list, alpha)
    # print(avg_weights)

    new_model = MVN4TrimNet(num_classes=6)
    new_model.load_state_dict(avg_weights)
    print(new_model)
