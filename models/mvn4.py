from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F


MODEL_SPECS = {
    "stage1": {"block_name": "convbn", "num_blocks": 1, "block_specs": [[3, 32, 3, 2]]},
    "stage2": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [[32, 32, 3, 2], [32, 32, 1, 1]],
    },
    "stage3": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [[32, 96, 3, 2], [96, 64, 1, 1]],
    },
    "stage4": {
        "block_name": "uib",
        "num_blocks": 4,
        "block_specs": [
            [64, 96, 5, 5, True, 2, 2],
            [96, 96, 0, 3, True, 1, 2],
            [96, 96, 0, 3, True, 1, 2],
            [96, 96, 3, 0, True, 1, 2],
        ],
    },
    "stage5": {
        "block_name": "uib",
        "num_blocks": 4,
        "block_specs": [
            [96, 128, 5, 5, True, 2, 2],
            [128, 128, 5, 3, True, 1, 2],
            [128, 128, 0, 3, True, 1, 2],
            [128, 128, 0, 3, True, 1, 2],
        ],
    },
    "stage6": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [[128, 960, 1, 1], [960, 1280, 1, 1]],
    },
}


GATE_SPEC = {
    "stage1": {"block_name": "convbn", "num_blocks": 1, "block_specs": [[3, 32, 3, 2]]},
    "stage2": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [[32, 32, 3, 2], [32, 32, 1, 1]],
    },
    "stage3": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [[32, 96, 3, 2], [96, 64, 1, 1]],
    },
}


def make_divisible(
    value: float,
    divisor: int,
    min_value: Optional[float] = None,
    round_down_protect: bool = True,
) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def conv_2d(
    inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True
):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.add_module(
        "conv",
        nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups),
    )
    if norm:
        conv.add_module("BatchNorm2d", nn.BatchNorm2d(oup))
    if act:
        conv.add_module("Activation", nn.ReLU6())
    return conv


class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        start_dw_kernel_size,
        middle_dw_kernel_size,
        middle_dw_downsample,
        stride,
        expand_ratio,
    ):
        super().__init__()
        # Starting depthwise conv.
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not middle_dw_downsample else 1

            self._start_dw_ = conv_2d(
                inp,
                inp,
                kernel_size=start_dw_kernel_size,
                stride=stride_,
                groups=inp,
                act=False,
            )
        # Expansion with 1x1 convs.
        expand_filters = make_divisible(inp * expand_ratio, 8)
        self._expand_conv = conv_2d(inp, expand_filters, kernel_size=1)
        # Middle depthwise conv.
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1

            self._middle_dw = conv_2d(
                expand_filters,
                expand_filters,
                kernel_size=middle_dw_kernel_size,
                stride=stride_,
                groups=expand_filters,
            )
        # Projection with 1x1 convs.
        self._proj_conv = conv_2d(
            expand_filters, oup, kernel_size=1, stride=1, act=False
        )

        # Ending depthwise conv.
        # this not used
        # _end_dw_kernel_size = 0
        # self._end_dw = conv_2d(oup, oup, kernel_size=_end_dw_kernel_size, stride=stride, groups=inp, act=False)

    def forward(self, x):
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
            # print("_start_dw_", x.shape)
        x = self._expand_conv(x)
        # print("_expand_conv", x.shape)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
            # print("_middle_dw", x.shape)
        x = self._proj_conv(x)
        # print("_proj_conv", x.shape)
        # print(x.shape)
        return x


def build_blocks(layer_spec):
    if not layer_spec.get("block_name"):
        return nn.Sequential()
    block_names = layer_spec["block_name"]
    layers = nn.Sequential()
    if block_names == "convbn":
        schema_ = ["inp", "oup", "kernel_size", "stride"]
        for i in range(layer_spec["num_blocks"]):
            args = dict(zip(schema_, layer_spec["block_specs"][i]))
            layers.add_module(f"convbn_{i}", conv_2d(**args))
    elif block_names == "uib":
        schema_ = [
            "inp",
            "oup",
            "start_dw_kernel_size",
            "middle_dw_kernel_size",
            "middle_dw_downsample",
            "stride",
            "expand_ratio",
        ]
        for i in range(layer_spec["num_blocks"]):
            args = dict(zip(schema_, layer_spec["block_specs"][i]))
            layers.add_module(f"uib_{i}", UniversalInvertedBottleneckBlock(**args))
    else:
        raise NotImplementedError
    return layers


class MVN4TrimNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.spec = MODEL_SPECS
        # conv0
        self.conv0 = build_blocks(self.spec["stage1"])
        # layer1
        self.layer1 = build_blocks(self.spec["stage2"])
        # layer2
        self.layer2 = build_blocks(self.spec["stage3"])
        # layer3
        self.layer3 = build_blocks(self.spec["stage4"])
        # layer4
        self.layer4 = build_blocks(self.spec["stage5"])
        # layer5
        self.layer5 = build_blocks(self.spec["stage6"])
        # fc
        self.fc = nn.Linear(1280, num_classes)

        self._kaiming_init()

    def _kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x5 = F.adaptive_avg_pool2d(x5, 1)
        out = self.fc(x5.view(x5.size(0), -1))
        return out


class GateTrimNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.origin_spec = GATE_SPEC

        self.stage1 = build_blocks(self.origin_spec["stage1"])
        self.stage2 = build_blocks(self.origin_spec["stage2"])
        self.stage3 = build_blocks(self.origin_spec["stage3"])
        self.fc = nn.Linear(64, 1)
        self.activation = nn.Sigmoid()

        self._kaiming_init()

    def _kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.activation(x)
        return x
