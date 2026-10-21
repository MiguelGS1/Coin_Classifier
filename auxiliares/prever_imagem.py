import argparse

import numpy as np
import tensorflow as tf

from config import DATASET_DIR
from dataset_loader import build_single_image_tensor

# Mostra o que um modelo já treinado (.keras) acha de UMA foto (ou de algumas fotos). Não treina nada.
# Diferente do predict.py, aqui você escolhe o modelo (ex.: o balanço de branco + cutout).


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prevê a moeda de uma ou mais fotos com o modelo escolhido.")
    parser.add_argument("--modelo", required=True, help="Arquivo .keras")
    parser.add_argument("--imagem", required=True, nargs="+", help="Uma ou mais fotos")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())
    model = tf.keras.models.load_model(args.modelo)
    for path in args.imagem:
        probs = model(build_single_image_tensor(path), training=False).numpy()[0]
        print(f"\nFoto: {path}")
        print(f"  Resposta do modelo: {class_names[int(np.argmax(probs))]}")
        for i in np.argsort(probs)[::-1]:
            print(f"    {class_names[i]:<30} {probs[i] * 100:6.2f}%")


if __name__ == "__main__":
    main()
