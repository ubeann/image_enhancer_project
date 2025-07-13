import glob
import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from typing import Tuple

def load_dataset(blur_dir: str, sharp_dir: str, image_size=(256, 256)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the dataset from the specified directories.

    Args:
        blur_dir (str): Directory containing blurred images.
        sharp_dir (str): Directory containing sharp images.
        image_size (tuple): Size to which images will be resized (default is (256, 256)).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of blurred and sharp images.
    """

    # Get list of all PNG files in the specified directories, sorted to maintain order
    blur_paths = sorted(glob.glob(f"{blur_dir}/*.png"))
    sharp_paths = sorted(glob.glob(f"{sharp_dir}/*.png"))

    # X for blurred images, Y for sharp images
    X, Y = [], []

    # Iterate through the file paths of blurred and sharp images
    for blur_path, sharp_path in zip(blur_paths, sharp_paths):
        # Read blurred and sharp images from disk
        blur = cv2.imread(blur_path)
        sharp = cv2.imread(sharp_path)

        # Resize images to the target size
        blur = cv2.resize(blur, image_size)
        sharp = cv2.resize(sharp, image_size)

        # Normalize pixel values to [0, 1] and append to respective lists
        X.append(blur / 255.0)
        Y.append(sharp / 255.0)

    # Convert lists to NumPy arrays with float32 precision
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def augment(x, y) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply random data augmentations to input and label images.

    Args:
        x (tf.Tensor): Input (blurred) image.
        y (tf.Tensor): Label (sharp) image.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Augmented input and label images.
    """
    # Apply random horizontal flip to both input and label
    x = tf.image.random_flip_left_right(x)
    y = tf.image.random_flip_left_right(y)

    # Apply random brightness adjustment to both input and label
    x = tf.image.random_brightness(x, 0.1)
    y = tf.image.random_brightness(y, 0.1)

    return x, y

def get_dataset(base_path: str, batch_size=8, val_split=0.2) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Load dataset, split into training and validation sets, apply augmentations,
    and prepare TensorFlow dataloaders.

    Args:
        blur_dir (str): Directory with blurred images.
        sharp_dir (str): Directory with sharp images.
        batch_size (int): Number of samples per batch.
        val_split (float): Fraction of data to use for validation.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation datasets.
    """
    # Define directories for blurred and sharp images
    blur_dir = os.path.join(base_path, "blur/images")
    sharp_dir = os.path.join(base_path, "sharp/images")

    # Load blurred and sharp image datasets
    X, Y = load_dataset(blur_dir, sharp_dir)

    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_split, random_state=42)

    # Create TensorFlow Dataset for training
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_ds = train_ds.map(augment)                     # Apply data augmentation
    train_ds = train_ds.batch(batch_size)                         # Group into batches
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)   # Optimize performance with prefetching

    # Create TensorFlow Dataset for validation (no augmentation)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(batch_size)

    # Return the training and validation datasets
    return train_ds, val_ds
