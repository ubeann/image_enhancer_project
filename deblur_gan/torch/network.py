# Import necessary libraries
import functools
import torch.nn as nn
from typing import Callable

# Import custom modules
from .fpn import FPNMobileNet

def get_norm_layer(norm_type: str) -> Callable[..., nn.Module]:
    """
    Returns a normalization layer constructor based on the specified type.

    Args:
        norm_type (str): Type of normalization. Options are:
                         'batch' - for Batch Normalization
                         'instance' - for Instance Normalization

    Returns:
        Callable[..., nn.Module]: A function (partial constructor) that returns
                                  an instance of the specified normalization layer
                                  when called with the required arguments (e.g., number of channels).

    Raises:
        NotImplementedError: If the provided norm_type is not recognized.
    """

    # Batch Normalization with affine parameters (learnable scale and shift)
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)

    # Instance Normalization without affine parameters but with running stats
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)

    # Raise error for unsupported normalization types
    else:
        raise NotImplementedError(f'Normalization layer [{norm_type}] is not found')

    # Return the normalization layer constructor
    return norm_layer

def get_generator(norm_type: str) -> nn.DataParallel:
    """
    Constructs and returns a generator model wrapped in DataParallel for multi-GPU training.

    Args:
        norm_type (str): Type of normalization to use in the generator. Common options: 'batch', 'instance', etc.

    Returns:
        nn.DataParallel: A generator model (`FPNMobileNet`) wrapped with `nn.DataParallel`.
    """
    # Create the generator model with the specified normalization layer
    model_g = FPNMobileNet(norm_layer=get_norm_layer(norm_type=norm_type))

    # Wrap the model with DataParallel for multi-GPU support
    return nn.DataParallel(model_g)


