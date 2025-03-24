# FL-MoE-for-CAN
> 这是用于CAN异常检测的 FedMoE 联邦学习框架

Baseline 模型为从 MobileNetv4 中剪枝得到的 UIBNet 作为基线模型（主要是检测快）；
整体氛围三部分
- GateModel：混合专家系统的门控网络，输入为 UIBNet 的前三层卷积特征提取层，在输出层拉平为线性，最后聚合到一个神经元上，用 sigmoid 激活；
- GlobalModel 和 LocalModel 都是 UIBNet，前者使用全局数据集进行训练聚合，后者使用本地私有数据集进行本地训练；


后面可以从 IID 分布和狄利克雷Non-IID分布下各自进行一轮简单测试；

# References
- 原版的代码：https://github.com/edvinli/federated-learning-mixture
- KTH 的老哥魔改了一下投的 FL-IJCNN-2022，不过似乎有集群：https://github.com/martisak/federated-learning-mixture