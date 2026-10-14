import argparse

import numpy as np
import tensorflow as tf

from config import DATASET_DIR
from dataset_loader import build_single_image_tensor

# Igual ao prever_imagem.py, mas com vários modelos (ex.: as 3 sementes): mostra a resposta de cada modelo e a do
# ENSEMBLE (classe com a maior média das probabilidades, sem pesos). Para UMA foto ou algumas. Não treina nada.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prevê a moeda de uma ou mais fotos com o ensemble de vários modelos.")
    parser.add_argument("--modelos", nargs="+", required=True, help="Arquivos .keras (ex.: as 3 sementes)")
    parser.add_argument("--nomes", nargs="+", required=True, help="Um nome curto por modelo (ex.: s42 s1 s2)")
    parser.add_argument("--imagem", nargs="+", required=True, help="Uma ou mais fotos")
    return parser.parse_args()


def short(name: str) -> str:
    return name.split()[0]


def main() -> None:
    args = parse_args()
    if len(args.modelos) != len(args.nomes):
        raise SystemExit("Passe um nome para cada modelo (--nomes).")
    class_names = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())
    models = [tf.keras.models.load_model(m) for m in args.modelos]
    header = "".join(f"{short(c):>12}" for c in class_names)
    for path in args.imagem:
        image = build_single_image_tensor(path)
        probs = np.stack([m(image, training=False).numpy()[0] for m in models])
        mean = probs.mean(axis=0)
        print(f"\nFoto: {path}")
        print(f"  {'':<10}{header}   resposta")
        for nome, p in zip(args.nomes, probs):
            print(f"  {nome:<10}" + "".join(f"{x * 100:>11.1f}%" for x in p) + f"   {short(class_names[int(p.argmax())])}")
        print(f"  {'ENSEMBLE':<10}" + "".join(f"{x * 100:>11.1f}%" for x in mean)
              + f"   {class_names[int(mean.argmax())]}")


if __name__ == "__main__":
    main()
