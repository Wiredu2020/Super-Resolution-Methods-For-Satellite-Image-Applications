"""Data pipeline.

Two ways to build a dataset:
  - make_dataset(hr_dir, lr_dir, ...)   real, pre-paired LR/HR patches
    (use this for WorldStrat / SEN2VENuS / OLI2MSI, or for the RSA data
    once real-sensor LR is available instead of synthetic degradation).
  - make_dataset(hr_dir, lr_dir=None, ...)  synthetic degradation, matching
    the bicubic/bilinear downsampling used for SRCNN/EDSR/SRGAN in this
    thesis, kept as a fallback / for direct comparison against the
    existing models on the exact same data.

Swapping real for synthetic pairs is a one-argument change, not a
model or training-loop change.
"""

import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
IMG_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg")


def _list_images(directory):
    files = []
    for pattern in IMG_EXTENSIONS:
        files.extend(tf.io.gfile.glob(f"{directory}/{pattern}"))
    return sorted(files)


def _decode(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # -> [0, 1]
    return img


def _augment(lr, hr):
    if tf.random.uniform(()) > 0.5:
        lr = tf.image.flip_left_right(lr)
        hr = tf.image.flip_left_right(hr)
    k = tf.random.uniform((), maxval=4, dtype=tf.int32)
    lr = tf.image.rot90(lr, k)
    hr = tf.image.rot90(hr, k)
    return lr, hr


def make_dataset(
    hr_dir,
    lr_dir=None,
    scale=2,
    hr_size=512,
    batch_size=8,
    shuffle=True,
    augment=True,
    degradation_method="bicubic",
):
    """Build a (lr, hr) tf.data.Dataset.

    hr_dir: directory of HR patches (all resized/cropped to hr_size).
    lr_dir: directory of real LR patches paired by sorted filename order
        with hr_dir. If None, LR is generated on the fly from HR via
        `degradation_method` downsampling by `scale` (synthetic, matches
        the rest of this thesis's data pipeline).
    """
    hr_files = _list_images(hr_dir)
    if not hr_files:
        raise FileNotFoundError(f"No images found in {hr_dir}")

    if lr_dir is not None:
        lr_files = _list_images(lr_dir)
        if len(lr_files) != len(hr_files):
            raise ValueError(
                f"lr_dir has {len(lr_files)} images, hr_dir has {len(hr_files)}; "
                "they must be paired 1:1 in sorted order."
            )
        ds = tf.data.Dataset.from_tensor_slices((lr_files, hr_files))

        def load_pair(lr_path, hr_path):
            lr = _decode(lr_path)
            hr = _decode(hr_path)
            hr = tf.image.resize(hr, (hr_size, hr_size))
            lr = tf.image.resize(lr, (hr_size // scale, hr_size // scale))
            return lr, hr

    else:
        ds = tf.data.Dataset.from_tensor_slices(hr_files)

        def load_pair(hr_path):
            hr = _decode(hr_path)
            hr = tf.image.resize(hr, (hr_size, hr_size))
            lr = tf.image.resize(
                hr, (hr_size // scale, hr_size // scale), method=degradation_method
            )
            return lr, hr

    ds = ds.map(load_pair, num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(hr_files), 2000))
    if augment:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def split_train_val(hr_dir, val_fraction=0.1, seed=42):
    """Utility to pre-split a flat HR directory's filenames for reproducible
    train/val partitions before pointing make_dataset at each half."""
    import random

    files = _list_images(hr_dir)
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_fraction))
    return files[n_val:], files[:n_val]
