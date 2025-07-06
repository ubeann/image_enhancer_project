import os
import tensorflow as tf

# Function to build the Zero-DCE (Deep Curve Estimation) neural network architecture
def build_dce_net() -> tf.keras.Model:
    """
    Builds the Zero-DCE model architecture for image enhancement.

    Returns:
        tf.keras.Model: A Keras model instance representing the Zero-DCE architecture.
    """

    # Define the input layer for images with 3 channels (RGB) and dynamic width/height
    input_img = tf.keras.Input(shape=[None, None, 3])

    # Series of convolutional layers with ReLU activation
    conv1 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(input_img)
    conv2 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(conv1)
    conv3 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(conv2)
    conv4 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(conv3)

    # Concatenate the output of conv4 and conv3 along the channel dimension
    int_con1 = tf.keras.layers.Concatenate(axis=-1)([conv4, conv3])
    conv5 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(int_con1)

    # Concatenate the output of conv5 and conv2
    int_con2 = tf.keras.layers.Concatenate(axis=-1)([conv5, conv2])
    conv6 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(int_con2)

    # Concatenate the output of conv6 and conv1
    int_con3 = tf.keras.layers.Concatenate(axis=-1)([conv6, conv1])

    # Final output layer with 24 filters and tanh activation
    # The tanh activation helps in learning the curve parameters in the range [-1, 1]
    x_r = tf.keras.layers.Conv2D(24, 3, activation="tanh", padding="same")(int_con3)

    # Build and return the Keras model
    model = tf.keras.Model(inputs=input_img, outputs=x_r)
    return model

# Function to load pretrained weights into the model
def load_zero_dce_weights(model_path: str) -> tf.keras.Model:
    """
    Loads the Zero-DCE model architecture and its pretrained weights.

    Args:
        model_path (str): Path to the pretrained weights file.

    Returns:
        tf.keras.Model: A Keras model instance with loaded weights.
    """

    # Build the model architecture
    model = build_dce_net()

    # Check if the specified weights file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    # Load the pretrained weights
    model.load_weights(model_path)

    return model
