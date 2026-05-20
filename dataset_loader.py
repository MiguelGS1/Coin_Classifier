from pathlib import Path
from typing import Tuple

import tensorflow as tf

from config import BATCH_SIZE, DATASET_DIR, IMAGE_SIZE, SEED, VALIDATION_SPLIT

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_dataset_structure(dataset_dir: Path = DATASET_DIR) -> None:
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Pasta do dataset não encontrada: {dataset_dir}\n"
            "Crie algo como dataset/raw/1_real_brasil, dataset/raw/1_euro, etc."
        )

    class_dirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
    if len(class_dirs) < 2:
        raise ValueError(
            "Você precisa de pelo menos 2 classes conhecidas para o treino multiclasse."
        )

    empty_classes = []
    for class_dir in class_dirs:
        has_images = any(
            p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
            for p in class_dir.rglob("*")
        )
        if not has_images:
            empty_classes.append(class_dir.name)

    if empty_classes:
        raise ValueError(
            "As seguintes classes não possuem imagens válidas: "
            + ", ".join(empty_classes)
        )


def build_datasets(
    dataset_dir: Path = DATASET_DIR,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    validate_dataset_structure(dataset_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    return train_ds, val_ds, class_names


def build_single_image_tensor(image_path: str) -> tf.Tensor:
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=0)
    return image
