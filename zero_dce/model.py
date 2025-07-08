import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict

# === Utils === #
def load_model_from_weights(weight_path: str) -> tf.keras.Model:
    """
    Loads the ZeroDCE model and applies pretrained weights.

    Args:
        weight_path (str): Path to the .h5 weights file.

    Returns:
        tf.keras.Model: The loaded and compiled ZeroDCE model.
    """
    model = ZeroDCE()
    model.load_weights(weight_path)
    return model

# === DCE Network === #
def build_dce_net() -> keras.Model:
    """
    Builds the Zero-DCE model architecture.
    Returns:
        keras.Model: The Zero-DCE model.
    """

    # Input layer for images of any size with 3 color channels (RGB)
    input_img = keras.Input(shape=[None, None, 3])
    conv1 = layers.Conv2D(32, 3, activation="relu", padding="same")(input_img)
    conv2 = layers.Conv2D(32, 3, activation="relu", padding="same")(conv1)
    conv3 = layers.Conv2D(32, 3, activation="relu", padding="same")(conv2)
    conv4 = layers.Conv2D(32, 3, activation="relu", padding="same")(conv3)

    # Concatenate layers to create a U-Net style architecture
    concat1 = layers.Concatenate()([conv4, conv3])
    conv5 = layers.Conv2D(32, 3, activation="relu", padding="same")(concat1)

    # Further concatenate with previous layers
    concat2 = layers.Concatenate()([conv5, conv2])
    conv6 = layers.Conv2D(32, 3, activation="relu", padding="same")(concat2)

    # Final concatenation with the first layer
    concat3 = layers.Concatenate()([conv6, conv1])
    x_r = layers.Conv2D(24, 3, activation="tanh", padding="same")(concat3)

    # Reshape the output to match the expected output shape
    return keras.Model(inputs=input_img, outputs=x_r)

# === Losses ===
def color_constancy_loss(x: tf.Tensor) -> tf.Tensor:
    """
    Computes the color constancy loss based on the mean RGB values.
    Args:
        x (tf.Tensor): Input tensor of shape (batch_size, height, width, 3).
    Returns:
        tf.Tensor: Color constancy loss value.
    """
    mean_rgb = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
    mr, mg, mb = mean_rgb[..., 0], mean_rgb[..., 1], mean_rgb[..., 2]
    return tf.sqrt(tf.square(mr - mg) + tf.square(mr - mb) + tf.square(mg - mb))

def exposure_loss(x: tf.Tensor, mean_val: float = 0.6) -> tf.Tensor:
    """
    Computes the exposure loss based on the average luminance of the image.
    Args:
        x (tf.Tensor): Input tensor of shape (batch_size, height, width, 3).
        mean_val (float): Target mean luminance value.
    Returns:
        tf.Tensor: Exposure loss value.
    """
    gray = tf.reduce_mean(x, axis=3, keepdims=True)
    pool = tf.nn.avg_pool2d(gray, ksize=16, strides=16, padding="VALID")
    return tf.reduce_mean(tf.square(pool - mean_val))

def illumination_smoothness_loss(x: tf.Tensor) -> tf.Tensor:
    """
    Computes the illumination smoothness loss based on the variance of pixel differences.
    Args:
        x (tf.Tensor): Input tensor of shape (batch_size, height, width, 3).
    Returns:
        tf.Tensor: Illumination smoothness loss value.
    """
    h_var = tf.reduce_sum(tf.square(x[:, 1:, :, :] - x[:, :-1, :, :]))
    w_var = tf.reduce_sum(tf.square(x[:, :, 1:, :] - x[:, :, :-1, :]))
    return (h_var + w_var)

# === Spatial Consistency Loss ===
class SpatialConsistencyLoss:
    """
    Computes the spatial consistency loss between enhanced and original images.
    This loss ensures that the gradients of the enhanced image match those of the original image.
    """

    def __init__(self):
        """
        Initializes the SpatialConsistencyLoss instance.
        This class does not require any parameters for initialization.
        """
        pass

    def __call__(self, enhanced: tf.Tensor, org: tf.Tensor) -> tf.Tensor:
        """
        Computes the spatial consistency loss between the enhanced and original images.
        Args:
            enhanced (tf.Tensor): Enhanced image tensor of shape (batch_size, height, width, 3).
            org (tf.Tensor): Original image tensor of shape (batch_size, height, width, 3).
        Returns:
            tf.Tensor: Spatial consistency loss value.
        """
        def gradient_map(img: tf.Tensor):
            gray = tf.image.rgb_to_grayscale(img)
            grad_x = gray[:, :, 1:, :] - gray[:, :, :-1, :]
            grad_y = gray[:, 1:, :, :] - gray[:, :-1, :, :]
            return grad_x, grad_y

        grad_x_enh, grad_y_enh = gradient_map(enhanced)
        grad_x_org, grad_y_org = gradient_map(org)

        loss_x = tf.reduce_mean(tf.square(grad_x_enh - grad_x_org))
        loss_y = tf.reduce_mean(tf.square(grad_y_enh - grad_y_org))
        return loss_x + loss_y

# === Main Model ===
class ZeroDCE(keras.Model):
    """
    ZeroDCE model for low-light image enhancement.
    This model uses a deep convolutional network to enhance low-light images
    by learning a set of curve parameters that adjust the pixel values.
    """

    def __init__(self, **kwargs):
        """
        Initializes the ZeroDCE model.
        Args:
            **kwargs: Additional keyword arguments for the keras.Model.
        """
        super().__init__(**kwargs)
        self.dce_model = build_dce_net()

    def compile(self, learning_rate: float, **kwargs):
        """
        Compiles the ZeroDCE model with an optimizer and loss functions.
        Args:
            learning_rate (float): Learning rate for the optimizer.
            **kwargs: Additional keyword arguments for the keras.Model.
        """
        super().compile(**kwargs)
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.spatial_constancy_loss = SpatialConsistencyLoss()

    def call(self, data: tf.Tensor) -> tf.Tensor:
        """
        Forward pass of the ZeroDCE model.
        Args:
            data (tf.Tensor): Input tensor of shape (batch_size, height, width, 3).
        Returns:
            tf.Tensor: Output tensor of shape (batch_size, height, width, 24) representing the curve parameters for enhancement.
        """
        return self.get_enhanced_image(data, self.dce_model(data))

    def get_enhanced_image(self, data: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
        """
        Applies the Zero-DCE enhancement formula using the curve parameters from the model output.
        Args:
            data (tf.Tensor): Input image tensor of shape (batch_size, height, width, 3) normalized to [0, 1].
            output (tf.Tensor): Output tensor from the model of shape (batch_size, height, width, 24) representing curve parameters.
        Returns:
            tf.Tensor: Enhanced image tensor of shape (batch_size, height, width, 3).
        """
        r = [output[..., i*3:(i+1)*3] for i in range(8)]
        x = data
        for ri in r:
            x = x + ri * (tf.square(x) - x)
        return x

    def compute_losses(self, data: tf.Tensor, output: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Computes the various losses for the ZeroDCE model.
        Args:
            data (tf.Tensor): Input image tensor of shape (batch_size, height, width, 3).
            output (tf.Tensor): Output tensor from the model of shape (batch_size, height, width, 24) representing curve parameters.
        Returns:
            Dict[str, tf.Tensor]: Dictionary containing the total loss and individual loss components.
        """
        # Get the enhanced image using the model output
        enhanced = self.get_enhanced_image(data, output)
        loss_illum = illumination_smoothness_loss(output)
        loss_spatial = self.spatial_constancy_loss(enhanced, data)
        loss_color = tf.reduce_mean(color_constancy_loss(enhanced))
        loss_exposure = tf.reduce_mean(exposure_loss(enhanced))

        # Compute the total loss as a weighted sum of individual losses
        total_loss = (
            200 * loss_illum +
            loss_spatial +
            5 * loss_color +
            10 * loss_exposure
        )

        # Return a dictionary of losses
        return {
            "total_loss": total_loss,
            "illumination_smoothness": loss_illum,
            "spatial_constancy": loss_spatial,
            "color_constancy": loss_color,
            "exposure": loss_exposure
        }

    def train_step(self, data):
        """
        Performs a single training step on the ZeroDCE model.
        Args:
            data (tf.Tensor): Input image tensor of shape (batch_size, height, width, 3).
        Returns:
            Dict[str, tf.Tensor]: Dictionary containing the losses computed during the training step.
        """
        # Wrap the forward pass and loss computation in a GradientTape
        with tf.GradientTape() as tape:
            output = self.dce_model(data)
            losses = self.compute_losses(data, output)

        # Compute gradients and apply them using the optimizer
        grads = tape.gradient(losses["total_loss"], self.dce_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.dce_model.trainable_weights))

        # Return the losses for logging
        return losses

    def test_step(self, data):
        """
        Performs a single test step on the ZeroDCE model.
        Args:
            data (tf.Tensor): Input image tensor of shape (batch_size, height, width, 3).
        Returns:
            Dict[str, tf.Tensor]: Dictionary containing the losses computed during the test step.
        """
        output = self.dce_model(data)
        return self.compute_losses(data, output)

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        """
        Saves the weights of the ZeroDCE model to a file.
        Args:
            filepath (str): Path to the file where weights will be saved.
            overwrite (bool): Whether to overwrite existing weights file.
            save_format (str): Format to save the weights, e.g., 'h5' or 'tf'.
            options: Additional options for saving.
        """
        self.dce_model.save_weights(filepath, overwrite=overwrite, save_format=save_format, options=options)

    def load_weights(self, filepath):
        """
        Loads the weights of the ZeroDCE model from a file.
        Args:
            filepath (str): Path to the file from which weights will be loaded.
            by_name (bool): Whether to load weights by name.
            skip_mismatch (bool): Whether to skip loading weights that do not match the model.
            options: Additional options for loading.
        """
        self.dce_model.load_weights(filepath=filepath)
