"""LSMAN architecture rebuild + weight loading.

The architecture JSON exported from the training notebook cannot be deserialised
directly: its Lambda layers were saved without an explicit ``output_shape`` and
Keras 3 cannot infer one when the spatial dimensions are dynamic.  We therefore
rebuild the graph from source (identical construction order to the notebook's
``build_lsman``) and take the layer *names* from the exported JSON, in order, so
that name-based weight loading still matches.
"""

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

FILTERS = 48
BLOCKS = 6

THESIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH_JSON = os.path.join(THESIS, "results", "lsman_x{s}_architecture.json")
WEIGHTS = os.path.join(THESIS, "weights", "lsman_x{s}_best.weights.h5")


class _Names:
    """Hands out the layer names recorded in the exported architecture JSON."""

    def __init__(self, path):
        cfg = json.load(open(path))
        self.queue = [(l["class_name"], l["config"]["name"])
                      for l in cfg["config"]["layers"]]
        self.i = 0

    def __call__(self, cls):
        got_cls, name = self.queue[self.i]
        if got_cls != cls:
            raise RuntimeError(
                f"layer {self.i}: rebuilt a {cls} but the export has {got_cls}. "
                "Construction order has drifted from the notebook.")
        self.i += 1
        return name


def _se_block(x, nm, reduction=8):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling2D(name=nm("GlobalAveragePooling2D"))(x)
    s = layers.Dense(max(ch // reduction, 4), activation="relu", name=nm("Dense"))(s)
    s = layers.Dense(ch, activation="sigmoid", name=nm("Dense"))(s)
    s = layers.Reshape((1, 1, ch), name=nm("Reshape"))(s)
    return layers.Multiply(name=nm("Multiply"))([x, s])


def _spatial_attention_block(x, nm):
    keep = lambda t: (t[0], t[1], t[2], 1)
    avg = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
                        output_shape=keep, name=nm("Lambda"))(x)
    mx = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
                       output_shape=keep, name=nm("Lambda"))(x)
    concat = layers.Concatenate(axis=-1, name=nm("Concatenate"))([avg, mx])
    attn = layers.Conv2D(1, 7, padding="same", activation="sigmoid",
                         name=nm("Conv2D"))(concat)
    return layers.Multiply(name=nm("Multiply"))([x, attn])


def _dual_attention_block(x, filters, nm):
    inp = x
    y = layers.Conv2D(filters, 3, padding="same", activation="relu", name=nm("Conv2D"))(x)
    y = layers.Conv2D(filters, 3, padding="same", name=nm("Conv2D"))(y)
    y = _se_block(y, nm)
    y = _spatial_attention_block(y, nm)
    return layers.Add(name=nm("Add"))([inp, y])


def build_lsman(scale, filters=FILTERS, num_blocks=BLOCKS, names=None,
                input_shape=(None, None, 3)):
    """``input_shape`` is fixed only when exporting to a runtime that requires
    static shapes, such as TensorFlow Lite; training and evaluation use the
    default dynamic shape."""
    nm = names if names is not None else (lambda cls: None)

    inp = layers.Input(shape=input_shape, name=nm("InputLayer"))
    f0 = layers.Conv2D(filters, 3, padding="same", name=nm("Conv2D"))(inp)

    x = f0
    block_outputs = []
    for _ in range(num_blocks):
        x = _dual_attention_block(x, filters, nm)
        block_outputs.append(x)

    fused = layers.Concatenate(name=nm("Concatenate"))(block_outputs + [f0])
    fused = layers.Conv2D(filters, 1, padding="same", name=nm("Conv2D"))(fused)
    x = layers.Add(name=nm("Add"))([f0, fused])

    x = layers.Conv2D(filters * (scale ** 2), 3, padding="same", name=nm("Conv2D"))(x)
    x = layers.Lambda(lambda t: tf.nn.depth_to_space(t, scale),
                      output_shape=lambda s: (s[0], None, None, s[3] // scale ** 2),
                      name=nm("Lambda"))(x)
    out = layers.Conv2D(3, 3, padding="same", activation="sigmoid", name=nm("Conv2D"))(x)
    return keras.Model(inp, out, name=f"LSMAN_x{scale}")


def load_lsman(scale, input_shape=(None, None, 3)):
    """Rebuild LSMAN at ``scale`` and load the trained weights."""
    names = _Names(ARCH_JSON.format(s=scale))
    model = build_lsman(scale, names=names, input_shape=input_shape)
    model.load_weights(WEIGHTS.format(s=scale))
    return model


if __name__ == "__main__":
    import numpy as np

    for s in (2, 4):
        m = load_lsman(s)
        y = m.predict(np.random.rand(1, 64, 64, 3).astype("float32"), verbose=0)
        print(f"x{s}: {m.count_params():,} params | (1,64,64,3) -> {y.shape} | "
              f"range [{y.min():.3f}, {y.max():.3f}]")
