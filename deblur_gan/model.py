# deblur_gan/model.py

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, load_model
from typing import Tuple

def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Custom loss function combining Mean Squared Error (MSE) and Structural Similarity Index Measure (SSIM).

    Args:
        y_true (tf.Tensor): Ground truth images, shape [batch_size, height, width, channels].
        y_pred (tf.Tensor): Predicted images, same shape as y_true.

    Returns:
        tf.Tensor: Scalar tensor representing the combined loss.
    """
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return mse_loss + ssim_loss

def build_unet(input_shape: Tuple[int, int, int] = (256, 256, 3)) -> Model:
    """
    Builds a U-Net model for image-to-image tasks like segmentation or image restoration.

    Args:
        input_shape (Tuple[int, int, int]): Shape of the input images, e.g., (height, width, channels).

    Returns:
        Model: A Keras Model instance representing the U-Net.
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)

    def conv_block(x: Layer, filters: int) -> Layer:
        """
        Applies two convolutional layers followed by batch normalization.

        Args:
            x (Layer): Input tensor.
            filters (int): Number of filters for the convolutional layers.

        Returns:
            Layer: Output tensor after convolutions and batch normalization.
        """
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)  # First conv layer
        x = layers.BatchNormalization()(x)                                   # Normalizes the activations
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)  # Second conv layer
        x = layers.BatchNormalization()(x)                                   # Normalizes again
        return x

    def encoder(x: Layer, filters: int) -> Tuple[Layer, Layer]:
        """
        Encoder block: applies conv_block and max pooling.

        Args:
            x (Layer): Input tensor.
            filters (int): Number of filters for conv_block.

        Returns:
            Tuple[Layer, Layer]: Tuple of (conv output for skip connection, pooled output for next layer).
        """
        c = conv_block(x, filters)            # Convolutional feature extraction
        p = layers.MaxPooling2D((2, 2))(c)    # Downsamples the features
        return c, p

    def decoder(x: Layer, skip: Layer, filters: int) -> Layer:
        """
        Decoder block: upsampling followed by conv_block with skip connection.

        Args:
            x (Layer): Input tensor from the previous decoder or bottleneck.
            skip (Layer): Corresponding encoder output for skip connection.
            filters (int): Number of filters for conv_block.

        Returns:
            Layer: Output tensor after upsampling and feature fusion.
        """
        u = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)  # Upsamples the input
        u = layers.Concatenate()([u, skip])                                   # Concatenates with skip connection
        c = conv_block(u, filters)                                          # Refines combined features
        return c

    # Encoder
    c1, p1 = encoder(inputs, 64)
    c2, p2 = encoder(p1, 128)
    c3, p3 = encoder(p2, 256)
    c4, p4 = encoder(p3, 512)

    # Bottleneck
    b1 = conv_block(p4, 1024)

    # Decoder
    d1 = decoder(b1, c4, 512)
    d2 = decoder(d1, c3, 256)
    d3 = decoder(d2, c2, 128)
    d4 = decoder(d3, c1, 64)

    # Output
    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(d4)

    # Create model and return
    model = models.Model(inputs, outputs)
    return model

def load_deblurgan_model(model_path: str) -> tf.keras.Model:
    """
    Load the DeblurGAN model from the specified path.
    Args:
        model_path: Path to the saved model file.
    Returns:
        Loaded Keras model.
    """
    return load_model(
        model_path,
        custom_objects={"combined_loss": combined_loss},
        compile=False
    )
