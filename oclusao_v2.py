import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import ARTIFACTS_DIR, BASE_DIR, BATCH_SIZE, DATASET_DIR, IMAGE_SIZE
from io_utils import save_json
from occlusion_test import TESTS, region_masks, transform

# Mesmo teste de oclusão e cor do occlusion_test.py (mesmas regiões e mesmo cinza), mas no TESTE da divisão v2
# (séries nunca vistas) e para um ou mais modelos. Não altera o occlusion_test.py; só importa as funções dele.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teste de oclusão e cor no teste da divisão v2.")
    parser.add_argument("--modelos", nargs="+", required=True, help="Arquivos .keras (um ou mais)")
    parser.add_argument("--nomes", nargs="+", required=True, help="Um nome curto por modelo (ex.: baseline lpr)")
    parser.add_argument("--pasta", default=str(BASE_DIR / "dataset" / "split_series_v2" / "teste"))
    parser.add_argument("--saida", default=str(ARTIFACTS_DIR / "experiments" / "oclusao_v2"))
    return parser.parse_args()


def short(name: str) -> str:
    return name.split(",")[0]


def main() -> None:
    args = parse_args()
    if len(args.modelos) != len(args.nomes):
        raise SystemExit("Passe um nome para cada modelo (--nomes).")
    out_dir = Path(args.saida)
    if out_dir.exists():
        raise SystemExit(f"A pasta {out_dir} já existe. Use outra --saida.")
    class_names = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())
    test_ds = tf.keras.utils.image_dataset_from_directory(
        args.pasta, labels="inferred", label_mode="categorical", class_names=class_names,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False)
    masks = region_masks()

    results = {}
    for nome, path in zip(args.nomes, args.modelos):
        model = tf.keras.models.load_model(path)
        results[nome] = {}
        for test, label in TESTS.items():
            print(f"{nome}: {label}...")
            y_true, y_pred = [], []
            for images, labels in test_ds:
                probs = model(transform(images, test, masks), training=False).numpy()
                y_true.append(np.argmax(labels.numpy(), axis=1))
                y_pred.append(np.argmax(probs, axis=1))
            y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
            matrix = np.zeros((len(class_names),) * 2, dtype=int)
            np.add.at(matrix, (y_true, y_pred), 1)
            recall = {c: float(matrix[i, i] / matrix[i].sum()) for i, c in enumerate(class_names)}
            results[nome][test] = {"acuracia": float((y_true == y_pred).mean()),
                                   "balanceada": float(np.mean(list(recall.values()))),
                                   "acuracia_por_classe": recall, "matriz_confusao": matrix.tolist()}

    out_dir.mkdir(parents=True)
    save_json({"modelos": dict(zip(args.nomes, args.modelos)), "pasta": args.pasta, "classes": class_names,
               "resultados": results}, out_dir / "resultados_oclusao_v2.json")

    print(f"\nOclusão e cor no TESTE v2 ({int(np.sum(results[args.nomes[0]]['original']['matriz_confusao']))} imagens)")
    print(f"{'modelo':<10}{'teste':<18}" + "".join(f"{short(c):>12}" for c in class_names) + f"{'Geral':>9}{'Balanc.':>9}")
    for nome in args.nomes:
        for test, label in TESTS.items():
            r = results[nome][test]
            print(f"{nome:<10}{label:<18}" + "".join(f"{r['acuracia_por_classe'][c] * 100:>11.1f}%" for c in class_names)
                  + f"{r['acuracia'] * 100:>8.1f}%{r['balanceada'] * 100:>8.1f}%")
        print()
    print("Matrizes (linhas = verdadeira; colunas = " + ", ".join(short(c) for c in class_names) + "):")
    for nome in args.nomes:
        for test, label in TESTS.items():
            print(f"  {nome:<10}{label:<18}{results[nome][test]['matriz_confusao']}")
    print(f"\nResultados salvos em: {out_dir}")


if __name__ == "__main__":
    main()
