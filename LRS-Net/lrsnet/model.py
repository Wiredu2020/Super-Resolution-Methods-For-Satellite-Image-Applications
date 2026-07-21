"""LRS-Net: Lightweight Remote-Sensing Super-Resolution Network.

Same functional-API / custom train_step style as the EDSR implementation
elsewhere in this repo, so results are directly comparable. Two changes
vs. that baseline keep the parameter count down and fix the color-shift
failure mode seen in the MSRCNN experiment:

  1. Depthwise-separable convolutions inside each block instead of full
     3x3 convolutions (~8-9x fewer weights per block at the same width).
  2. A global residual connection: the network predicts a correction on
     top of a bilinearly-upsampled copy of the input, with a linear
     (unactivated) output layer, instead of learning absolute pixel
     values through a squashing activation.
"""

import tensorflow as tf
from tensorflow.keras import layers


class LRSNetModel(tf.keras.Model):
    """Custom train_step, mirrors EDSRModel in Models/EDSR/EDSR_main.ipynb."""

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


def channel_attention(inputs, filters, reduction=8, name=None):
    """Contrast-aware channel attention (mean + std pooling -> tiny MLP gate).

    Near-parameter-free (a couple hundred weights) but recovers accuracy
    lost from shrinking the backbone width/depth. See IMDN/RFDN.
    """
    avg_pool = layers.GlobalAveragePooling2D(keepdims=True)(inputs)
    std_pool = layers.Lambda(
        lambda x: tf.math.reduce_std(x, axis=[1, 2], keepdims=True)
    )(inputs)
    pooled = layers.Add()([avg_pool, std_pool])

    x = layers.Conv2D(max(filters // reduction, 4), 1, activation="relu")(pooled)
    x = layers.Conv2D(filters, 1, activation="sigmoid")(x)
    return layers.Multiply(name=name)([inputs, x])


def lrs_block(inputs, filters):
    """Depthwise-separable residual block with channel attention."""
    x = layers.SeparableConv2D(filters, 3, padding="same", activation="relu")(inputs)
    x = layers.SeparableConv2D(filters, 3, padding="same")(x)
    x = channel_attention(x, filters)
    x = layers.Add()([inputs, x])
    return x


def upsample(inputs, filters, factor=2):
    """Sub-pixel convolution upsampling (same pattern as EDSR's Upsampling)."""
    x = layers.Conv2D(filters * (factor ** 2), 3, padding="same")(inputs)
    x = layers.Lambda(lambda t: tf.nn.depth_to_space(t, block_size=factor))(x)
    return x


def _upsample_skip(inputs, factor):
    """Coarse base for the global residual skip. Bilinear, not bicubic:
    XLA has no GPU kernel for ResizeBicubic, so bicubic here crashes
    model.fit() under Keras 3's default XLA-compiled GPU training step.
    Bilinear has an XLA GPU kernel and is refined by the learned residual
    anyway, so there's no meaningful quality cost.
    """
    def resize(x):
        shape = tf.shape(x)
        new_size = shape[1:3] * factor
        return tf.image.resize(x, new_size, method="bilinear")

    return layers.Lambda(resize)(inputs)


def build_lrsnet(num_filters=32, num_blocks=6, scale=2, reduction=8):
    """Build LRS-Net.

    num_filters=32, num_blocks=6 targets ~150K-400K parameters, well
    below EDSR-16 (~1.5M) and SRGAN's generator (~1.5M), for comparison
    on the parameter/PSNR Pareto plot.
    """
    input_layer = layers.Input(shape=(None, None, 3))

    shallow = layers.Conv2D(num_filters, 3, padding="same")(input_layer)
    x = shallow
    for _ in range(num_blocks):
        x = lrs_block(x, num_filters)
    x = layers.SeparableConv2D(num_filters, 3, padding="same")(x)
    x = layers.Add()([shallow, x])

    x = upsample(x, num_filters, factor=scale)
    residual = layers.Conv2D(3, 3, padding="same", name="sr_residual")(x)

    base = _upsample_skip(input_layer, scale)
    output_layer = layers.Add(name="sr_output")([base, residual])

    return LRSNetModel(input_layer, output_layer, name="LRS-Net")
