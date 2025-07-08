import tensorflow as tf
import numpy as np
from PIL import Image
from .model import load_model

# Load the pretrained Zero-DCE model once at module load time
zero_dce_model = load_model("models/zero_dce.weights.h5")

# Function to apply enhancement using the learned curve parameters
def get_enhanced_image(data: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
    """
    Applies the Zero-DCE enhancement formula multiple times using the curve parameters.

    Args:
        data: Input image tensor with shape (1, H, W, 3) normalized to [0, 1]
        output: Output from the model with shape (1, H, W, 24) representing curve parameters

    Returns:
        Tensor representing the enhanced image
    """
    # Split the 24-channel output into 8 sets of 3-channel tensors (for RGB)
    r = [output[:, :, :, i:i+3] for i in range(0, 24, 3)]

    x = data  # Start with the original input image
    for ri in r:
        # Apply the enhancement function iteratively:
        # x = x + r * (x^2 - x)
        x = x + ri * (tf.square(x) - x)

    return x  # Final enhanced image tensor

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
    # Load the image and convert it to RGB mode
    image = Image.open(image_path).convert("RGB")

    # Convert image to float32 numpy array and normalize to [0, 1]
    input_array = tf.keras.utils.img_to_array(image) / 255.0

    # Add batch dimension to make it shape (1, H, W, 3)
    input_array = np.expand_dims(input_array, axis=0)

    # Run the image through the model to get enhancement curves
    output_tensor = zero_dce_model(input_array)

    # Apply the learned curve parameters to enhance the image
    enhanced = get_enhanced_image(input_array, output_tensor)

    # Convert the result back to a uint8 image (scale from [0, 1] to [0, 255])
    enhanced_image = tf.cast(enhanced[0] * 255.0, dtype=tf.uint8).numpy()

    # Create a PIL Image from the result
    result = Image.fromarray(enhanced_image)

    # Save the image if a save path is provided
    if save_path:
        result.save(save_path)

    return result  # Return the enhanced image as a PIL object
