"""Evaluation metrics and model-cost accounting for the params/PSNR Pareto plot."""

import tensorflow as tf


def log10(x):
    return tf.math.log(x) / tf.math.log(tf.constant(10, dtype=x.dtype))


def PSNR(y_true, y_pred):
    """Same formula as Models/EDSR/EDSR_main.ipynb, kept identical for
    numbers to be directly comparable across models."""
    mse = tf.reduce_mean((y_true - y_pred) ** 2)
    return 20 * log10(1 / (mse ** 0.5))


def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def count_params(model):
    trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
    non_trainable = sum(int(tf.size(w)) for w in model.non_trainable_weights)
    return {
        "trainable": trainable,
        "non_trainable": non_trainable,
        "total": trainable + non_trainable,
    }


def estimate_flops(model, input_shape=(1, 256, 256, 3)):
    """Approximate multiply-accumulate count (MACs) via TF's graph profiler.

    Good enough for a relative params/FLOPs/PSNR comparison table; not a
    substitute for a hardware-measured benchmark.
    """
    concrete = tf.function(lambda x: model(x)).get_concrete_function(
        tf.TensorSpec(input_shape, tf.float32)
    )
    frozen = tf.python.framework.convert_to_constants.convert_variables_to_constants_v2(
        concrete
    )
    graph_info = tf.compat.v1.profiler.profile(
        graph=frozen.graph,
        options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation(),
    )
    return graph_info.total_float_ops
