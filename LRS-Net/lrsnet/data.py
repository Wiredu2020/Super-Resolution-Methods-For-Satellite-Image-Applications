"""Data pipeline.

The RSA-derived Processed_Data only has flat HR/ and LR/ folders (2202
matched-by-filename pairs, no train/val/test split) -- splitting happens
here, in code, on the file-path lists, not by physically copying files
into subfolders. That also works against read-only mounts (a Kaggle
"Add Input" dataset, for example).

Two ways to get (hr_files, lr_files) to feed in:
  - list_pairs(hr_dir, lr_dir)   real, pre-paired LR/HR patches (RSA data,
    or a real-pair dataset like WorldStrat via worldstrat.find_pairs)
  - list_pairs(hr_dir, lr_dir=None)   HR only; make_dataset then generates
    LR on the fly via synthetic bicubic/bilinear downsampling, matching
    the degradation used for SRCNN/EDSR/SRGAN elsewhere in this thesis.

Swapping real for synthetic pairs, or RSA data for WorldStrat, is a
different list of paths going into the same split_pairs/make_dataset
calls -- not a model or training-loop change.
"""

import json
import os
import random

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


def list_pairs(hr_dir, lr_dir=None):
    """List HR (and optionally matching LR) file paths from flat directories.

    Pairing is by identical filename, not just sort order + count, so a
    mismatch fails loudly instead of silently pairing the wrong images.
    """
    hr_files = _list_images(hr_dir)
    if not hr_files:
        raise FileNotFoundError(f"No images found in {hr_dir}")

    if lr_dir is None:
        return hr_files, None

    lr_files = _list_images(lr_dir)
    hr_names = {os.path.basename(f) for f in hr_files}
    lr_names = {os.path.basename(f) for f in lr_files}
    if hr_names != lr_names:
        missing = hr_names - lr_names
        extra = lr_names - hr_names
        raise ValueError(
            f"HR/LR filename mismatch: {len(missing)} HR files have no LR match, "
            f"{len(extra)} LR files have no HR match (e.g. {list(missing)[:3] or list(extra)[:3]})."
        )
    # lr_files re-derived from hr_dir's basenames so the two lists line up 1:1.
    lr_files = [os.path.join(lr_dir, os.path.basename(f)) for f in hr_files]
    return hr_files, lr_files


def split_pairs(hr_files, lr_files=None, val_fraction=0.1, test_fraction=0.1, seed=42):
    """Reproducible random train/val/test split over paired file lists.

    Defaults (80/10/10) match the split ratio WorldStrat's authors use,
    so RSA data and WorldStrat data are split the same way.
    """
    n = len(hr_files)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    n_val = max(1, int(n * val_fraction))
    n_test = max(1, int(n * test_fraction))
    val_idx = indices[:n_val]
    test_idx = indices[n_val:n_val + n_test]
    train_idx = indices[n_val + n_test:]

    def gather(files, idx):
        return [files[i] for i in idx] if files is not None else None

    return {
        "train": (gather(hr_files, train_idx), gather(lr_files, train_idx)),
        "val": (gather(hr_files, val_idx), gather(lr_files, val_idx)),
        "test": (gather(hr_files, test_idx), gather(lr_files, test_idx)),
    }


def make_dataset(
    hr_files,
    lr_files=None,
    scale=2,
    hr_size=512,
    batch_size=8,
    shuffle=True,
    augment=True,
    degradation_method="bicubic",
):
    """Build a (lr, hr) tf.data.Dataset from explicit file-path lists.

    hr_files: list of HR image paths.
    lr_files: list of real LR image paths, paired 1:1 with hr_files. If
        None, LR is generated on the fly from HR via `degradation_method`
        downsampling by `scale`.
    """
    if not hr_files:
        raise ValueError("hr_files is empty.")

    if lr_files is not None:
        if len(lr_files) != len(hr_files):
            raise ValueError(
                f"lr_files has {len(lr_files)} paths, hr_files has {len(hr_files)}."
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


def load_split_datasets(
    hr_dir,
    lr_dir=None,
    val_fraction=0.1,
    test_fraction=0.1,
    seed=42,
    save_split_json=None,
    **dataset_kwargs,
):
    """list_pairs + split_pairs + make_dataset for all three splits at once.

    Returns {"train": ds, "val": ds, "test": ds}. If save_split_json is
    given, writes the exact file lists per split there -- keep this so the
    same test split can be reloaded later for the cross-model comparison
    (params/PSNR Pareto plot, downstream-task eval) instead of re-splitting
    with drifted data.
    """
    hr_files, lr_files = list_pairs(hr_dir, lr_dir)
    splits = split_pairs(hr_files, lr_files, val_fraction, test_fraction, seed)

    if save_split_json:
        os.makedirs(os.path.dirname(save_split_json), exist_ok=True)
        with open(save_split_json, "w") as f:
            json.dump(
                {name: {"hr": hr, "lr": lr} for name, (hr, lr) in splits.items()},
                f,
                indent=2,
            )

    datasets = {}
    for name, (hr, lr) in splits.items():
        datasets[name] = make_dataset(
            hr,
            lr,
            shuffle=(name == "train"),
            augment=(name == "train"),
            **dataset_kwargs,
        )
    return datasets
