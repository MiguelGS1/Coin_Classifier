import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import ARTIFACTS_DIR, DATASET_DIR, MODEL_PATH
from dataset_loader import build_single_image_tensor
from io_utils import save_json

# Avalia um modelo em uma pasta de imagens organizada por classe (ex.: dataset/controle_eua).
# A pasta pode ter só algumas das classes (o controle_eua não tem Washington). Não treina nada.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia um modelo em uma pasta de imagens por classe.")
    parser.add_argument("--modelo", default=str(MODEL_PATH), help="Arquivo .keras (padrão: o modelo final)")
    parser.add_argument("--pasta", required=True, help="Pasta com uma subpasta por classe")
    parser.add_argument("--nome", required=True, help="Nome do resultado (ex.: v2_controle)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = ARTIFACTS_DIR / "experiments" / "avaliacoes" / f"{args.nome}.json"
    if out_path.exists():
        raise SystemExit(f"Já existe um resultado em {out_path}. Use outro --nome.")
    # Mesma ordem de classes usada no treino (pastas do dataset/raw em ordem alfabética)
    class_names = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())
    folder = Path(args.pasta)

    model = tf.keras.models.load_model(args.modelo)
    n = len(class_names)
    matrix = np.zeros((n, n), dtype=int)
    for class_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        if class_dir.name not in class_names:
            raise SystemExit(f"Pasta '{class_dir.name}' não é uma das classes do modelo: {class_names}")
        true = class_names.index(class_dir.name)
        paths = sorted(str(p) for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        for start in range(0, len(paths), 32):
            images = tf.concat([build_single_image_tensor(p) for p in paths[start:start + 32]], axis=0)
            for pred in np.argmax(model(images, training=False).numpy(), axis=1):
                matrix[true, pred] += 1

    present = matrix.sum(axis=1) > 0
    recall = {c: float(matrix[i, i] / matrix[i].sum()) for i, c in enumerate(class_names) if present[i]}
    total, hits = int(matrix.sum()), int(matrix.diagonal().sum())
    result = {"modelo": args.modelo, "pasta": str(folder), "imagens": total, "acertos": hits,
              "acuracia": hits / total, "balanceada": float(np.mean(list(recall.values()))),
              "revocacao": recall, "classes": class_names, "matriz_confusao": matrix.tolist()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(result, out_path)

    print(f"\nModelo: {Path(args.modelo).name}   Pasta: {folder}")
    print(f"Acurácia: {hits}/{total} = {result['acuracia'] * 100:.1f}%   "
          f"Balanceada (classes presentes): {result['balanceada'] * 100:.1f}%")
    print("Matriz de confusão (linhas = verdadeira; colunas = " + ", ".join(c.split()[0] for c in class_names) + ")")
    for i, c in enumerate(class_names):
        if present[i]:
            print(f"  {c.split()[0]:<11} {matrix[i].tolist()}   acerto {recall[c] * 100:.1f}%")
    print(f"\nResultado salvo em: {out_path}")


if __name__ == "__main__":
    main()
