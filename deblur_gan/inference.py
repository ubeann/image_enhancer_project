# Import necessary libraries
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from sklearn.metrics import mean_squared_error
from tensorflow.keras.utils import img_to_array

# Import custom modules
from .model import load_deblurgan_model

# Load once (module level)
deblurgan_model = load_deblurgan_model("models/deblurgan.keras")

def deblur_image(image_path: str, save_path: str = None) -> tuple:
    """
    Loads an image, processes it, passes it through the DeblurGAN model,
    and returns the deblurred image.

    Args:
        image_path (str): Path to the input blurred image.
        save_path (str, optional): If provided, the deblurred image will be saved to this path.

    Returns:
        Image.Image: The deblurred image as a PIL Image object.
        float: Mean Squared Error between original and deblurred images.
        float: Peak Signal-to-Noise Ratio between original and deblurred images.
    """

    # Load original image and preserve its unprocessed version
    original = Image.open(image_path).convert("RGB")
    original_size = original.size
    original_uint8 = np.array(original)

    # Resize to model input
    resized = original.resize((256, 256))
    arr = img_to_array(resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Predict
    pred = deblurgan_model.predict(arr)[0]
    pred = np.clip(pred, 0, 1) * 255.0
    result = Image.fromarray(pred.astype("uint8"))

    # Resize back to original
    result = result.resize(original_size)
    enhanced_uint8 = np.array(result)

    # Metrics (ensure same shape)
    if original_uint8.shape == enhanced_uint8.shape:
        mse_val = mean_squared_error(original_uint8.flatten(), enhanced_uint8.flatten())
        psnr_val = peak_signal_noise_ratio(original_uint8, enhanced_uint8, data_range=255)
    else:
        mse_val, psnr_val = None, None  # Could also raise warning

    # Save the result if a save path is provided
    if save_path:
        result.save(save_path)

    # Return the result image and metrics
    return result, mse_val, psnr_val
