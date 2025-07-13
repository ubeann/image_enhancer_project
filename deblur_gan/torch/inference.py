# Import necessary libraries
import torch
import numpy as np
from torchvision import transforms
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from PIL import Image

# Import custom modules
from .loader import load_deblurganv2_model

# Load once (module level)
deblurgan_model = load_deblurganv2_model("models/deblurganv2_mobilenet.pth")

def deblur_image(image_path: str, save_path: str = None) -> tuple:
    """
    Loads an image, processes it, passes it through the DeblurGAN (PyTorch) model,
    and returns the deblurred image along with MSE and PSNR metrics.

    Args:
        image_path (str): Path to the input blurred image.
        save_path (str, optional): If provided, the deblurred image will be saved to this path.

    Returns:
        PIL.Image.Image: The deblurred image.
        float: Mean Squared Error (MSE) between original and deblurred image.
        float: Peak Signal-to-Noise Ratio (PSNR) between original and deblurred image.
    """

    # Load the input image using PIL
    original = Image.open(image_path)

    # Convert image to RGBA if it has a transparency palette, then to RGB
    if original.mode == "P" and "transparency" in original.info:
        original = original.convert("RGBA")
    original = original.convert("RGB")

    # Save original image size and pixel values for later use
    original_size = original.size
    original_uint8 = np.array(original)

    # Resize image to the fixed input size expected by the model (e.g., 256x256)
    resized = original.resize((256, 256))

    # Preprocess: Convert image to tensor (C, H, W) with pixel values in [0, 1]
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(resized).unsqueeze(0)  # Add batch dimension (1, 3, H, W)

    # Inference: Feed the input tensor through the DeblurGAN model
    with torch.no_grad():
        pred_tensor = deblurgan_model(input_tensor).squeeze(0).clamp(0, 1)

    # Postprocess: Convert the output tensor back to a uint8 image
    pred_np = pred_tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    result = Image.fromarray(pred_np)

    # Resize output image back to the original size
    result = result.resize(original_size)

    # Compute evaluation metrics (MSE and PSNR) if image shapes match
    enhanced_uint8 = np.array(result)
    if original_uint8.shape == enhanced_uint8.shape:
        mse_val = mean_squared_error(original_uint8, enhanced_uint8)
        psnr_val = peak_signal_noise_ratio(original_uint8, enhanced_uint8, data_range=255)
    else:
        mse_val, psnr_val = None, None  # Cannot compute metrics due to shape mismatch

    # Save the deblurred image to disk if a save path is provided
    if save_path:
        result.save(save_path)

    # Return the deblurred image and evaluation metrics
    return result, mse_val, psnr_val
