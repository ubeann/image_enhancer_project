import tensorflow as tf
import numpy as np
from PIL import Image
from .model import load_model_from_weights

# Load the pretrained Zero-DCE model once at module load time
zero_dce_model = load_model_from_weights("models/zero_dce.weights.h5")

# Main function to enhance an image from a file
def enhance_image(image_path: str, save_path: str = None) -> Image.Image:
    """
    Loads an image, enhances it using Zero-DCE, and optionally saves the result.

    Args:
        image_path: Path to the input image file
        save_path: Optional path to save the enhanced image

    Returns:
        A PIL Image object of the enhanced image
    """
    # Load and normalize the image
    image = Image.open(image_path).convert("RGB")
    input_array = tf.keras.utils.img_to_array(image) / 255.0
    input_array = np.expand_dims(input_array, axis=0)  # shape: [1, H, W, 3]

    # Get enhanced image directly from model
    enhanced = zero_dce_model(input_array)  # shape: [1, H, W, 3]

    # Convert to uint8 image
    enhanced_image = tf.cast(enhanced[0] * 255.0, dtype=tf.uint8).numpy()
    result = Image.fromarray(enhanced_image)

    # Save the image if a save path is provided
    if save_path:
        result.save(save_path)

    return result  # Return the enhanced image as a PIL object
