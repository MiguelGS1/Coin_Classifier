import hashlib
from collections import defaultdict
from pathlib import Path

import tensorflow as tf

from config import BATCH_SIZE, DATASET_DIR, IMAGE_SIZE, SEED, VALIDATION_SPLIT


def split_files(subset: str) -> list[str]:
    # Mesma divisão do build_datasets (mesma pasta, semente e proporção), só para obter os caminhos
    ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR, validation_split=VALIDATION_SPLIT, subset=subset, seed=SEED,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="categorical",
    )
    return list(ds.file_paths)


def file_hash(path: str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main() -> None:
    train, val = split_files("training"), split_files("validation")
    side = {p: "treino" for p in train} | {p: "validação" for p in val}

    groups = defaultdict(list)
    for path in train + val:
        groups[file_hash(path)].append(path)
    duplicated = [g for g in groups.values() if len(g) > 1]

    print(f"\nGrupos de arquivos idênticos: {len(duplicated)} "
          f"({sum(len(g) for g in duplicated)} arquivos, {sum(len(g) - 1 for g in duplicated)} cópias a mais)")
    crossing = []
    for g in duplicated:
        sides = {side[p] for p in g}
        names = ", ".join(f"{Path(p).name} ({side[p]})" for p in g)
        print(f"  {'TREINO E VALIDAÇÃO' if len(sides) > 1 else 'só ' + sides.pop():<18}  {names}")
        if len({side[p] for p in g}) > 1:
            crossing += [p for p in g if side[p] == "validação"]

    print(f"\nImagens de validação que têm uma cópia idêntica no treino: {len(crossing)} de {len(val)} "
          f"({len(crossing) / len(val) * 100:.1f}%)")
    for p in crossing:
        print(f"  {Path(p).parent.name}/{Path(p).name}")


if __name__ == "__main__":
    main()
