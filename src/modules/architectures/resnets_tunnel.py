from itertools import chain

import torch
from torch import Tensor
from typing import Callable, List, Optional, Type, Union
import torch.nn as nn
from functools import partial

from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1


class BasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        self.skips = kwargs.pop("skips")
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if self.skips:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None and self.skips:
            identity = self.downsample(x)

        if self.skips:
            out += identity
        out = self.relu(out)

        return out


class Bottleneck(Bottleneck):
    def __init__(self, *args, **kwargs):
        self.skips = kwargs.pop("skips", True)
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if self.skips:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None and self.skips:
            identity = self.downsample(x)

        if self.skips:
            out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        width_scale: float = 1.0,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        skips: bool = True,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.scale_width = width_scale
        self.skips = skips

        self.inplanes = int(64 * width_scale)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 * width_scale * block.expansion), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:

        planes = int(planes * self.scale_width)

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                skips=self.skips,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    skips=self.skips,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
    def forward_generator(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for block in chain(self.layer1, self.layer2, self.layer3, self.layer4):
            x = block(x)
            x = yield x, None
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.fc(x)
        _ = yield None, y


def build_resnet(model_config, num_classes, dataset_name):

    backbone_type = model_config['backbone_type']
    only_features = model_config['only_features']
    batchnorm_layers = model_config['batchnorm_layers']
    width_scale = model_config['width_scale']
    skips = model_config['skips']
    modify_resnet = model_config['modify_resnet']
    
    
    modify_resnet = modify_resnet and (dataset_name == "cifar100" or dataset_name == "cifar10")

    # model = torchvision.models.__dict__[backbone_type](num_classes=num_classes)
    resnet = partial(
        ResNet, num_classes=num_classes, width_scale=width_scale, skips=skips
    )
    if not batchnorm_layers:
        resnet = partial(resnet, norm_layer=nn.Identity)
    match backbone_type:
        case "resnet18":
            model = resnet(BasicBlock, [2, 2, 2, 2])
        case "resnet34":
            model = resnet(BasicBlock, [3, 4, 6, 3])
        case "resnet50":
            model = resnet(Bottleneck, [3, 4, 6, 3])
        case "resnet101":
            model = resnet(Bottleneck, [3, 4, 23, 3])
        case "resnet152":
            model = resnet(Bottleneck, [3, 8, 36, 3])
        case _:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

    if modify_resnet:
        model.maxpool = torch.nn.Identity()
        model.conv1 = torch.nn.Conv2d(
            3, int(64 * width_scale), kernel_size=3, stride=1, padding=2, bias=False
        )
    if only_features:
        model.fc = torch.nn.Identity()

    if not batchnorm_layers:
        # turn off batch norm tracking stats and learning parameters
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.track_running_stats = False
                m.affine = False
                m.running_mean = None
                m.running_var = None

    renset_penultimate_layer_size = {
        "resnet18": int(512 * width_scale),
        "resnet34": int(512 * width_scale),
        "resnet50": int(2048 * width_scale),
        "resnet101": int(2048 * width_scale),
        "resnet152": int(2048 * width_scale),
    }
    model.penultimate_layer_size = renset_penultimate_layer_size[backbone_type]

    return model


def _forward_impl(self, x: Tensor) -> Tensor:
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)

    return x


if __name__ == "__main__":
    pass