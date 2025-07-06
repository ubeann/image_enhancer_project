import tensorflow as tf
import os
from glob import glob
from typing import List, Tuple

# Constants
IMAGE_SIZE: int = 256     # Size to which all images will be resized
BATCH_SIZE: int = 5       # Number of images per training batch

def load_data(image_path: tf.Tensor) -> tf.Tensor:
    """
    Reads and processes a single image from a file path.

    Args:
        image_path (tf.Tensor): File path to the image (as a scalar string tensor).

    Returns:
        tf.Tensor: Preprocessed image tensor normalized to [0, 1], shape (IMAGE_SIZE, IMAGE_SIZE, 3)
    """
    # Load image file
    image = tf.io.read_file(image_path)

    # Decode PNG to tensor with 3 channels
    image = tf.image.decode_png(image, channels=3)

    # Resize image
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])

    # Normalize pixel values to [0, 1]
    return image / 255.0


def data_generator(image_paths: List[str]) -> tf.data.Dataset:
    """
    Creates a tf.data.Dataset from a list of image paths.

    Args:
        image_paths (List[str]): List of image file paths.

    Returns:
        tf.data.Dataset: Dataset yielding batches of preprocessed images.
    """
    # Create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    # Load and preprocess images
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch for performance
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Return the dataset
    return dataset


def get_dataset(base_path: str, max_train: int = 300) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Loads and splits the LOL dataset into training and validation datasets.

    Args:
        base_path (str): Base directory path where the LOL dataset is stored.
        max_train (int): Maximum number of training samples to use.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation datasets.
    """
    # Load training images (low-light images)
    train_low = sorted(glob(os.path.join(base_path, "our485/low/*")))[:max_train]
    train_dataset = data_generator(train_low)

    # Load validation images (high-quality corresponding images)
    val_low = sorted(glob(os.path.join(base_path, "our485/high/*")))[max_train:]
    val_dataset = data_generator(val_low)

    return train_dataset, val_dataset
