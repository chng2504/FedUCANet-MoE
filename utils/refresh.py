import torch

try:
    import torch_npu
except ImportError:
    pass


def clear_cache(device: str):
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "npu":
        torch_npu.npu.empty_cache()
    else:
        pass
