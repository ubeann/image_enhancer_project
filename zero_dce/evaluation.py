import numpy as np

def calculate_psnr(original_image, processed_image):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    Args:
        original_image: The original image as a numpy array.
        processed_image: The processed image as a numpy array.
    Returns:
        PSNR value as a float.
    """
    mse = np.mean((original_image - processed_image) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel_value = 255.0  # Use 255 for standard 8-bit images
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

def calculate_mse(original_image, processed_image):
    """
    Calculate the Mean Squared Error (MSE) between two images.
    Args:
        original_image: The original image as a numpy array.
        processed_image: The processed image as a numpy array.
    Returns:
        MSE value as a float.
    """
    return np.mean((original_image - processed_image) ** 2)