# Import necessary libraries
import os
import torch
import torch.nn as nn

# Import custom modules
from .network import get_generator

def load_deblurganv2_model(weights_path: str, norm_type: str = 'instance') -> nn.DataParallel:
    """
    Loads the DeblurGANv2 model with pretrained MobileNetV2 backbone.

    Args:
        weights_path (str): Path to the pretrained model weights.
        norm_type (str): Type of normalization to use in the generator. Common options: 'batch', 'instance', etc.

    Returns:
        nn.DataParallel: The loaded and ready-to-use DeblurGANv2 model wrapped in DataParallel.

    Raises:
        FileNotFoundError: If the specified weights file does not exist.
    """
    # Check if the weights file exists
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ Model weights not found: {weights_path}")

    # Initialize the generator model (e.g., DeblurGANv2 with MobileNetV2 encoder)
    model = get_generator(weights_path=weights_path, norm_type=norm_type)

    # Set the model to evaluation mode (important for inference)
    model.eval()

    # Wrap model in DataParallel for multi-GPU compatibility (even if only one GPU is used)
    return nn.DataParallel(model)