from pathlib import Path
import tensorflow as tf

DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_BATCH_SIZE = 32


def load_dataset(
    data_dir: str | Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 42,
) -> tf.data.Dataset:
    return tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )


def scale_dataset(data: tf.data.Dataset) -> tf.data.Dataset:
    return data.map(
        lambda x, y: (
            tf.cast(x, tf.float32) / 255.0,
            y
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def split_dataset(
    data: tf.data.Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    total_batches = len(data)

    if total_batches < 3:
        raise ValueError(
            "The dataset needs at least 3 batches to create train/val/test splits."
        )

    train_size = max(1, int(total_batches * train_ratio))
    val_size = max(1, int(total_batches * val_ratio))

    if train_size + val_size >= total_batches:
        val_size = max(1, total_batches - train_size - 1)

    test_size = total_batches - train_size - val_size

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)

    return train, val, test