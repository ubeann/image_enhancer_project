# Import necessary libraries
import tensorflow as tf
import numpy as np
from PIL import Image

# Import custom modules
from .model import load_model_from_weights
from .evaluation import calculate_mse, calculate_psnr

# Load the pretrained Zero-DCE model once at module load time
zero_dce_model = load_model_from_weights("models/zero_dce.weights.h5")

# Main function to enhance an image from a file
def enhance_image(image_path: str, save_path: str = None) -> tuple:
    """
    Loads an image, enhances it using Zero-DCE, and optionally saves the result.

    Args:
        image_path: Path to the input image file
        save_path: Optional path to save the enhanced image

    Returns:
        result (PIL.Image.Image): The enhanced image as a PIL Image.
        mse (float): Mean Squared Error between input and output.
        psnr (float): Peak Signal-to-Noise Ratio between input and output.
    """
    # Load and normalize the image
    image = Image.open(image_path).convert("RGB")
    input_array = tf.keras.utils.img_to_array(image) / 255.0
    input_array = np.expand_dims(input_array, axis=0)  # shape: [1, H, W, 3]

    # Get enhanced image directly from model
    enhanced = zero_dce_model(input_array)  # shape: [1, H, W, 3]

    # Convert both original and enhanced images to uint8 [0, 255]
    original_uint8 = tf.cast(input_array[0] * 255.0, dtype=tf.uint8).numpy()
    enhanced_uint8 = tf.cast(enhanced[0] * 255.0, dtype=tf.uint8).numpy()

    # Calculate metrics
    mse = calculate_mse(original_uint8, enhanced_uint8)
    psnr = calculate_psnr(original_uint8, enhanced_uint8)

    # Convert the enhanced image back to PIL format
    result = Image.fromarray(enhanced_uint8)

    # Save the image if a save path is provided
    if save_path:
        result.save(save_path)

    # Return the result image and metrics
    return result, mse, psnr
