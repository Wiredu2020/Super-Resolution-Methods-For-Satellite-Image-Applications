"""Loss functions. Matches the SSIM-loss choice used for SRCNN/MSRCNN."""

import tensorflow as tf


def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def l1_ssim_loss(alpha=0.84):
    """Weighted L1 + SSIM loss, common in lightweight-SR literature (IMDN, RFDN).

    alpha weights the SSIM term; (1 - alpha) weights L1. Default follows
    the value commonly used in the efficient-SR literature.
    """

    def loss_fn(y_true, y_pred):
        l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
        ssim = ssim_loss(y_true, y_pred)
        return alpha * ssim + (1.0 - alpha) * l1

    loss_fn.__name__ = "l1_ssim_loss"
    return loss_fn
