import tensorflow as tf

def build_dce_net() -> tf.keras.Model:
    """
    Builds the Zero-DCE enhancement model.
    Returns:
        tf.keras.Model: A compiled Keras model with DCE architecture.
    """
    input_img = tf.keras.Input(shape=[None, None, 3])

    conv1 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(input_img)
    conv2 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(conv1)
    conv3 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(conv2)
    conv4 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(conv3)

    concat1 = tf.keras.layers.Concatenate()([conv4, conv3])
    conv5 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(concat1)

    concat2 = tf.keras.layers.Concatenate()([conv5, conv2])
    conv6 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(concat2)

    concat3 = tf.keras.layers.Concatenate()([conv6, conv1])
    x_r = tf.keras.layers.Conv2D(24, 3, activation="tanh", padding="same")(concat3)

    return tf.keras.Model(inputs=input_img, outputs=x_r)

def color_constancy_loss(x: tf.Tensor) -> tf.Tensor:
    mean_rgb = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
    mr, mg, mb = mean_rgb[..., 0], mean_rgb[..., 1], mean_rgb[..., 2]
    return tf.sqrt(tf.square(mr - mg) + tf.square(mr - mb) + tf.square(mg - mb))

def exposure_loss(x: tf.Tensor, mean_val: float = 0.6) -> tf.Tensor:
    gray = tf.reduce_mean(x, axis=3, keepdims=True)
    pool = tf.nn.avg_pool2d(gray, ksize=16, strides=16, padding="VALID")
    return tf.reduce_mean(tf.square(pool - mean_val))

def illumination_smoothness_loss(x: tf.Tensor) -> tf.Tensor:
    h_variation = tf.reduce_sum(tf.square(x[:, 1:, :, :] - x[:, :-1, :, :]))
    w_variation = tf.reduce_sum(tf.square(x[:, :, 1:, :] - x[:, :, :-1, :]))
    return h_variation + w_variation

class SpatialConsistencyLoss:
    def __call__(self, enhanced: tf.Tensor, org: tf.Tensor) -> tf.Tensor:
        def gradient_map(img: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            gray = tf.image.rgb_to_grayscale(img)
            grad_x = gray[:, :, 1:, :] - gray[:, :, :-1, :]
            grad_y = gray[:, 1:, :, :] - gray[:, :-1, :, :]
            return grad_x, grad_y

        grad_x_enh, grad_y_enh = gradient_map(enhanced)
        grad_x_org, grad_y_org = gradient_map(org)

        loss_x = tf.reduce_mean(tf.square(grad_x_enh - grad_x_org))
        loss_y = tf.reduce_mean(tf.square(grad_y_enh - grad_y_org))
        return loss_x + loss_y

class ZeroDCE(tf.keras.Model):
    def __init__(self):
        """
        Initializes the Zero-DCE model with custom loss functions.
        """
        super().__init__()
        self.dce_model = build_dce_net()
        self.spatial_constancy_loss = SpatialConsistencyLoss()

    def compile(self, learning_rate: float):
        """
        Compile the model with a custom optimizer.
        """
        super().compile()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def get_enhanced_image(self, data: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
        """
        Applies the learned enhancement curve (r) iteratively to the input.
        """
        r = [output[..., i*3:(i+1)*3] for i in range(8)]
        x = data
        for ri in r:
            x = x + ri * (tf.square(x) - x)
        return x

    def compute_losses(self, data: tf.Tensor, output: tf.Tensor) -> dict[str, tf.Tensor]:
        """
        Computes total loss and its components.
        """
        enhanced = self.get_enhanced_image(data, output)
        loss = {
            "illumination_smoothness": illumination_smoothness_loss(output),
            "spatial_constancy": self.spatial_constancy_loss(enhanced, data),
            "color_constancy": tf.reduce_mean(color_constancy_loss(enhanced)),
            "exposure": tf.reduce_mean(exposure_loss(enhanced)),
        }
        loss["total_loss"] = (
            200 * loss["illumination_smoothness"] +
            loss["spatial_constancy"] +
            5 * loss["color_constancy"] +
            10 * loss["exposure"]
        )
        return loss

    def train_step(self, data: tf.Tensor) -> dict[str, tf.Tensor]:
        """
        Custom training loop using GradientTape.
        """
        with tf.GradientTape() as tape:
            output = self.dce_model(data)
            losses = self.compute_losses(data, output)

        grads = tape.gradient(losses["total_loss"], self.dce_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.dce_model.trainable_weights))
        return losses

    def call(self, data: tf.Tensor) -> tf.Tensor:
        """
        Performs enhancement on input image(s).
        """
        return self.get_enhanced_image(data, self.dce_model(data))

