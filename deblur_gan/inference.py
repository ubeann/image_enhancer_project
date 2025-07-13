# Import necessary libraries
import numpy as np
from PIL import Image
from tensorflow.keras.utils import img_to_array

# Import custom modules
from .model import load_deblurgan_model

# Load once (module level)
deblurgan_model = load_deblurgan_model("models/deblurgan.keras")

def deblur_image(image_path: str, save_path: str = None) -> Image.Image:
    """
    Loads an image, processes it, passes it through the DeblurGAN model,
    and returns the deblurred image.

    Args:
        image_path (str): Path to the input blurred image.
        save_path (str, optional): If provided, the deblurred image will be saved to this path.

    Returns:
        Image.Image: The deblurred image as a PIL Image object.
    """

    # Open the input image and ensure it's in RGB format and get its original size
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    # Resize the image to 256x256, matching the model's expected input size
    image = image.resize((256, 256))

    # Convert the image to a NumPy array and normalize pixel values to [0, 1]
    arr = img_to_array(image) / 255.0

    # Add a batch dimension to the array: (1, height, width, channels)
    arr = np.expand_dims(arr, axis=0)

    # Run the image through the preloaded DeblurGAN model
    pred = deblurgan_model.predict(arr)[0]  # Get the first (and only) prediction

    # Clip the predicted pixel values to [0, 1] and scale back to [0, 255]
    pred = np.clip(pred, 0, 1) * 255.0

    # Convert the array back to a PIL Image
    result = Image.fromarray(pred.astype("uint8"))

    # Resize back to original
    result = result.resize(original_size)

    # Save the image if a save path is provided
    if save_path:
        result.save(save_path)

    # Return the resulting deblurred image
    return result

