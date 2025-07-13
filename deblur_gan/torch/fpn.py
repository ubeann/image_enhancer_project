# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Import custom modules
from model import MobileNetV2

class FPNHead(nn.Module):
    """
    A simple two-layer convolutional head used on top of Feature Pyramid Network (FPN) outputs.

    This module is typically used in object detection or segmentation models to process
    the multi-scale feature maps produced by an FPN into task-specific outputs.
    """

    def __init__(self, num_in: int, num_mid: int, num_out: int):
        """
        Initializes the FPNHead module.

        Args:
            num_in (int): Number of input channels from the FPN.
            num_mid (int): Number of intermediate channels after the first convolution.
            num_out (int): Number of output channels after the second convolution.
        """
        super().__init__()

        # First convolution layer: transforms input channels to intermediate channels
        self.block0 = nn.Conv2d(
            in_channels=num_in,
            out_channels=num_mid,
            kernel_size=3,
            padding=1,  # preserves spatial dimensions
            bias=False  # no bias term; typically used with BatchNorm (not included here)
        )

        # Second convolution layer: transforms intermediate channels to output channels
        self.block1 = nn.Conv2d(
            in_channels=num_mid,
            out_channels=num_out,
            kernel_size=3,
            padding=1,  # preserves spatial dimensions
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the FPNHead.

        Args:
            x (torch.Tensor): Input tensor of shape (N, num_in, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, num_out, H, W)
        """
        # Apply first convolution and ReLU activation
        x = F.relu(self.block0(x), inplace=True)

        # Apply second convolution and ReLU activation
        x = F.relu(self.block1(x), inplace=True)

        # Return the processed feature map
        return x

class FPNMobileNet(nn.Module):
    """
    A Feature Pyramid Network (FPN) with a MobileNetV2 backbone for image-to-image tasks.

    This model is designed to perform tasks such as image reconstruction, enhancement, or translation
    by combining multi-scale features extracted from the input image using an FPN-style architecture.
    """

    def __init__(
        self,
        norm_layer: nn.Module,
        output_ch: int = 3,
        num_filters: int = 64,
        num_filters_fpn: int = 128,
        pretrained: bool = True
    ):
        """
        Initializes the FPNMobileNet model.

        Args:
            norm_layer (nn.Module): A normalization layer constructor (e.g., BatchNorm2d or InstanceNorm2d).
            output_ch (int): Number of output channels (typically 3 for RGB).
            num_filters (int): Number of channels in each FPN head and intermediate layers.
            num_filters_fpn (int): Number of channels in the FPN backbone feature maps.
            pretrained (bool): If True, loads pretrained MobileNetV2 weights into the FPN backbone.
        """
        super().__init__()

        # Backbone feature extractor with FPN based on MobileNetV2
        self.fpn = FPN(
            num_filters=num_filters_fpn,
            norm_layer=norm_layer,
            pretrained=pretrained
        )

        # FPN heads for each feature scale (from coarse to fine resolution)
        self.head1 = FPNHead(num_filters_fpn, num_filters, num_filters)  # finest resolution
        self.head2 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head4 = FPNHead(num_filters_fpn, num_filters, num_filters)  # coarsest resolution

        # Feature fusion block to smooth and compress concatenated features
        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters),
            nn.ReLU()
        )

        # Additional smoothing block to reduce dimensionality and fuse with map0
        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
            norm_layer(num_filters // 2),
            nn.ReLU()
        )

        # Final output convolution layer to produce reconstructed image
        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)

    def unfreeze(self):
        """
        Unfreezes the FPN backbone for fine-tuning during training.
        Useful when training on new datasets or tasks.
        """
        self.fpn.unfreeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            torch.Tensor: Reconstructed output image tensor of shape (N, output_ch, H, W)
        """
        # Extract multi-scale feature maps from FPN
        map0, map1, map2, map3, map4 = self.fpn(x)

        # Process each feature map with a head and upsample to input resolution
        map4 = F.upsample(self.head4(map4), scale_factor=8, mode="nearest")  # coarsest
        map3 = F.upsample(self.head3(map3), scale_factor=4, mode="nearest")
        map2 = F.upsample(self.head2(map2), scale_factor=2, mode="nearest")
        map1 = F.upsample(self.head1(map1), scale_factor=1, mode="nearest")  # finest (no upsample)

        # Concatenate all processed maps and apply smoothing convolution
        combined = torch.cat([map4, map3, map2, map1], dim=1)
        smoothed = self.smooth(combined)

        # Upsample to match map0's resolution and fuse with map0
        smoothed = F.upsample(smoothed, scale_factor=2, mode="nearest")
        smoothed = self.smooth2(smoothed + map0)  # residual fusion with early feature map

        # Upsample to original image resolution
        smoothed = F.upsample(smoothed, scale_factor=2, mode="nearest")

        # Final reconstruction layer followed by tanh activation and residual connection
        final = self.final(smoothed)
        res = torch.tanh(final) + x  # residual learning: output = input + correction
        return torch.clamp(res, min=-1, max=1)  # constrain values to [-1, 1]

class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN) using a MobileNetV2 backbone.

    This module constructs a multi-scale feature pyramid from an input image using
    MobileNetV2 as the encoder and a top-down FPN architecture to generate semantically
    rich feature maps at different scales for tasks like segmentation or reconstruction.
    """

    def __init__(self, norm_layer: nn.Module, num_filters: int = 128, pretrained: bool = True):
        """
        Initializes the FPN module.

        Args:
            norm_layer (nn.Module): A callable that returns a normalization layer (e.g., BatchNorm2d).
            num_filters (int): Number of output channels for the FPN layers.
            pretrained (bool): If True, loads pretrained weights for the MobileNetV2 backbone.
        """
        super().__init__()

        # Initialize MobileNetV2 backbone
        net = MobileNetV2(n_class=1000)

        if pretrained:
            # Load pretrained weights from file (ensure file exists or handle exception)
            state_dict = torch.load('mobilenetv2.pth.tar')  # Use map_location='cpu' if needed
            net.load_state_dict(state_dict)

        # Full MobileNetV2 feature layers
        self.features = net.features

        # Divide MobileNetV2 into encoder stages for multi-scale features
        self.enc0 = nn.Sequential(*self.features[0:2])   # Output: 16 channels
        self.enc1 = nn.Sequential(*self.features[2:4])   # Output: 24 channels
        self.enc2 = nn.Sequential(*self.features[4:7])   # Output: 32 channels
        self.enc3 = nn.Sequential(*self.features[7:11])  # Output: 64 channels
        self.enc4 = nn.Sequential(*self.features[11:16]) # Output: 160 channels

        # Lateral connections: 1x1 convolutions to match channel dimensions across pyramid
        self.lateral4 = nn.Conv2d(160, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(64, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(32, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(24, num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(16, num_filters // 2, kernel_size=1, bias=False)  # Lower-res map

        # Top-down pathway: combines upsampled higher-level features with lateral ones
        self.td1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters),
            nn.ReLU(inplace=True)
        )
        self.td2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters),
            nn.ReLU(inplace=True)
        )
        self.td3 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters),
            nn.ReLU(inplace=True)
        )

        # Freeze MobileNetV2 backbone by default
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreezes the MobileNetV2 backbone layers for training or fine-tuning.
        """
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the FPN.

        Args:
            x (torch.Tensor): Input image tensor of shape (N, C, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - lateral0: Low-level features (used as residual or skip connection).
                - map1: Finest resolution FPN feature map.
                - map2: Mid-level resolution FPN feature map.
                - map3: Coarse resolution FPN feature map.
                - map4: Coarsest resolution FPN feature map.
        """
        # Bottom-up encoding: extract features from different layers
        enc0 = self.enc0(x)  # Feature map after first couple of layers
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Apply 1x1 lateral convolutions to unify channel dimensions
        lateral4 = self.lateral4(enc4)  # Top layer (deepest features)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)
        lateral0 = self.lateral0(enc0)  # Bottom layer (early features)

        # Top-down path: progressively combine and refine features
        map4 = lateral4  # Start from deepest feature
        map3 = self.td1(lateral3 + F.upsample(map4, scale_factor=2, mode="nearest"))  # merge with upsampled deeper map
        map2 = self.td2(lateral2 + F.upsample(map3, scale_factor=2, mode="nearest"))
        map1 = self.td3(lateral1 + F.upsample(map2, scale_factor=2, mode="nearest"))

        # Return the full pyramid of features
        return lateral0, map1, map2, map3, map4
