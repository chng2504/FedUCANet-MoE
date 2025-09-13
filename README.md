# FL-MoE-for-CAN
> This is a FedMoE federated learning framework for CAN anomaly detection

The baseline model is UIBNet, obtained by pruning from MobileNetv4;
The overall framework consists of three parts:
- GateModel: The gating network of the mixture-of-experts system. Its input is the first three convolutional feature extraction layers of UIBNet, which are flattened into a linear layer at the output and finally aggregated into a single neuron with sigmoid activation.
- Both GlobalModel and LocalModel are UIBNet. The former is trained and aggregated using the global dataset, while the latter performs local training using the local private dataset.


# References
- Original code: https://github.com/edvinli/federated-learning-mixture
