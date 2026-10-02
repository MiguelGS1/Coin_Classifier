import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import ARTIFACTS_DIR, BASE_DIR, BATCH_SIZE, CLASS_NAMES_PATH, IMAGE_SIZE, MODEL_PATH
from dataset_loader import build_datasets
from io_utils import load_json, save_json

CELULAR_DIR = BASE_DIR / "dataset" / "celular" / "teste_celular"
OUTPUT_DIR = ARTIFACTS_DIR / "experiments" / "celular"
REGISTRO = OUTPUT_DIR / "registro.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Testa um modelo nas fotos de celular e anota o resultado no registro.")
    parser.add_argument("--modelo", default=str(MODEL_PATH), help="Arquivo .keras do modelo (padrão: o modelo final)")
    parser.add_argument("--nome", default="atual", help="Nome que aparece no registro")
    return parser.parse_args()


def predict(model, dataset):
    y_true, y_pred = [], []
    for images, labels in dataset:
        y_pred.append(np.argmax(model(images, training=False).numpy(), axis=1))
        y_true.append(np.argmax(labels.numpy(), axis=1))
    return np.concatenate(y_true), np.concatenate(y_pred)


def short(name: str) -> str:
    return name.split()[0]


def testar(model, nome: str, class_names: list) -> dict:
    if not CELULAR_DIR.exists():
        raise SystemExit(f"Pasta não encontrada: {CELULAR_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validação: a mesma divisão 80/20 do treino (fotos no mesmo estilo do treino)
    _, val_ds, _ = build_datasets()
    val_true, val_pred = predict(model, val_ds)

    # Celular: class_names garante a mesma ordem de classes do modelo
    cel_ds = tf.keras.utils.image_dataset_from_directory(
        CELULAR_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        label_mode="categorical", class_names=class_names, shuffle=False,
    )
    y_true, y_pred = predict(model, cel_ds)

    n = len(class_names)
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    recalls = {class_names[i]: float(matrix[i, i] / matrix[i].sum()) for i in range(n)}
    result = {
        "nome": nome,
        "data": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "acuracia_validacao": float((val_true == val_pred).mean()),
        "acuracia_celular": float((y_true == y_pred).mean()),
        "acuracia_balanceada_celular": float(np.mean(list(recalls.values()))),
        "acertos_por_classe_celular": recalls,
        "imagens_por_classe_celular": {class_names[i]: int(matrix[i].sum()) for i in range(n)},
        "matriz_confusao_celular": matrix.tolist(),
    }
    save_json(result, OUTPUT_DIR / f"resultado_{nome}.json")

    # Uma linha por teste no registro (abre direto no Excel)
    new_file = not REGISTRO.exists()
    with open(REGISTRO, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        if new_file:
            writer.writerow(["nome", "data", "validacao", "celular", "celular_balanceada"]
                            + [f"celular_{short(c)}" for c in class_names])
        pct = lambda v: f"{v * 100:.1f}".replace(".", ",")
        writer.writerow([nome, result["data"], pct(result["acuracia_validacao"]), pct(result["acuracia_celular"]),
                         pct(result["acuracia_balanceada_celular"])] + [pct(recalls[c]) for c in class_names])

    print(f"\nResultado: {nome}")
    print(f"  Validação (mesmo estilo do treino): {result['acuracia_validacao'] * 100:.1f}%")
    print(f"  Fotos de celular ({len(y_true)} imagens): {result['acuracia_celular'] * 100:.1f}%"
          f"   | balanceada (média das 3 moedas): {result['acuracia_balanceada_celular'] * 100:.1f}%")
    for i, c in enumerate(class_names):
        print(f"    {c:<32} acertou {matrix[i, i]:>3} de {matrix[i].sum():>3} ({recalls[c] * 100:.1f}%)")
    print("  Para onde foram as previsões (linhas = moeda certa):")
    print("    " + " " * 14 + "".join(f"{short(c):>12}" for c in class_names))
    for i, c in enumerate(class_names):
        print(f"    {short(c):<14}" + "".join(f"{v:>12}" for v in matrix[i]))
    print(f"\nAnotado em: {REGISTRO}")
    return result


def main() -> None:
    args = parse_args()
    model = tf.keras.models.load_model(args.modelo)
    testar(model, args.nome, load_json(CLASS_NAMES_PATH))


if __name__ == "__main__":
    main()
