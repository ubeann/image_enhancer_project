import torch
import torch.nn as nn
import math
from typing import List

def conv_bn(inp: int, oup: int, stride: int) -> nn.Sequential:
    """
    Creates a 3x3 convolution followed by BatchNorm and ReLU6.

    Args:
        inp (int): Number of input channels.
        oup (int): Number of output channels.
        stride (int): Stride for the convolution.

    Returns:
        nn.Sequential: Sequential container with Conv2d, BatchNorm2d, and ReLU6.
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp: int, oup: int) -> nn.Sequential:
    """
    Creates a 1x1 convolution followed by BatchNorm and ReLU6.

    Args:
        inp (int): Number of input channels.
        oup (int): Number of output channels.

    Returns:
        nn.Sequential: Sequential container with 1x1 Conv2d, BatchNorm2d, and ReLU6.
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    """
    Inverted Residual Block used in MobileNetV2.

    This block can optionally expand the number of channels using a pointwise convolution,
    apply a depthwise convolution, and then project back using another pointwise convolution.
    """
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        """
        Args:
            inp (int): Number of input channels.
            oup (int): Number of output channels.
            stride (int): Stride of the depthwise convolution (1 or 2).
            expand_ratio (int): Expansion factor for input channels.
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2], "Stride must be 1 or 2."

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            # No expansion
            self.conv = nn.Sequential(
                # Depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Pointwise linear projection
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            # With expansion
            self.conv = nn.Sequential(
                # Pointwise expansion
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Pointwise linear projection
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2 architecture as described in:
    "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (https://arxiv.org/abs/1801.04381)
    """
    def __init__(self, n_class: int = 1000, input_size: int = 224, width_mult: float = 1.0):
        """
        Args:
            n_class (int): Number of output classes.
            input_size (int): Input image size (must be divisible by 32).
            width_mult (float): Width multiplier for scaling model width.
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        # Each tuple: (expansion_factor, output_channels, num_blocks, stride)
        interverted_residual_setting: List[List[int]] = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        assert input_size % 32 == 0, "Input size must be divisible by 32."

        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        # First layer
        self.features = [conv_bn(3, input_channel, stride=2)]

        # Inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # Final 1x1 conv layer
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))

        # Wrap as nn.Sequential
        self.features = nn.Sequential(*self.features)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        # Weight initialization
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean(3).mean(2)  # Global average pooling over H and W
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        """
        Initializes weights for Conv, BatchNorm, and Linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
