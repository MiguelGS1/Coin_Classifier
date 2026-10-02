import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import CLASS_NAMES_PATH, MODEL_PATH
from dataset_loader import build_single_image_tensor
from io_utils import load_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Testa o modelo original (3 classes) na pasta de controle.")
    parser.add_argument("--controle", default="dataset/controle_eua", help="Pasta com uma subpasta por classe")
    args = parser.parse_args()

    # Só lê o modelo original; nada é salvo
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = load_json(CLASS_NAMES_PATH)
    print(f"\nModelo original ({MODEL_PATH.name}) na pasta {args.controle}:")
    for folder in sorted(p for p in Path(args.controle).iterdir() if p.is_dir()):
        if folder.name not in class_names:
            print(f"  Pasta '{folder.name}' ignorada (não é o nome de uma classe)")
            continue
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        images = tf.concat([build_single_image_tensor(str(p)) for p in paths], axis=0)
        probs = model(images, training=False).numpy()
        pred = np.argmax(probs, axis=1)
        counts = ", ".join(f"{n.split(',')[0]} {int((pred == i).sum())}" for i, n in enumerate(class_names))
        print(f"  {folder.name:<32} acertos {(pred == class_names.index(folder.name)).mean() * 100:5.1f}%  "
              f"confiança média {probs.max(axis=1).mean():.2f}  ({len(paths)} imagens: {counts})")


if __name__ == "__main__":
    main()
