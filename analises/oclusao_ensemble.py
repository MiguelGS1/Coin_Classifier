import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import ARTIFACTS_DIR, BASE_DIR, BATCH_SIZE, DATASET_DIR, IMAGE_SIZE
from io_utils import save_json
from occlusion_test import TESTS, region_masks, transform

# Mesmo teste de oclusão e cor do oclusao_v2.py (mesmas regiões, mesmo cinza, mesmas imagens do TESTE v2), mas
# mostrando também o ENSEMBLE: em cada teste, a resposta é a classe com a maior média das probabilidades dos
# modelos (soft voting, igual ao avaliar_ensemble.py). Mostra cada modelo, a média ± desvio dos modelos e o ensemble,
# separados. Não treina nada; as imagens só são alteradas na hora da avaliação.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teste de oclusão e cor dos modelos e do ensemble (teste v2).")
    parser.add_argument("--modelos", nargs="+", required=True, help="Arquivos .keras (ex.: as 3 sementes)")
    parser.add_argument("--nomes", nargs="+", required=True, help="Um nome curto por modelo (ex.: s42 s1 s2)")
    parser.add_argument("--pasta", default=str(BASE_DIR / "dataset" / "split_series_v2" / "teste"))
    parser.add_argument("--saida", default=str(ARTIFACTS_DIR / "experiments" / "oclusao_ensemble"))
    return parser.parse_args()


def short(name: str) -> str:
    return name.split(",")[0].split()[0]


def summary(y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> dict:
    matrix = np.zeros((len(class_names),) * 2, dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    recall = {c: float(matrix[i, i] / matrix[i].sum()) for i, c in enumerate(class_names)}
    return {"acuracia": float((y_true == y_pred).mean()), "balanceada": float(np.mean(list(recall.values()))),
            "acuracia_por_classe": recall, "matriz_confusao": matrix.tolist()}


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
    y_true = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test_ds])
    masks = region_masks()

    probs = {t: [] for t in TESTS}  # teste -> [modelo] -> [foto, classe]
    for nome, path in zip(args.nomes, args.modelos):
        model = tf.keras.models.load_model(path)
        for test, label in TESTS.items():
            print(f"{nome}: {label}...", flush=True)
            probs[test].append(np.concatenate(
                [model(transform(images, test, masks), training=False).numpy() for images, _ in test_ds]))

    results = {}
    for test in TESTS:
        p = np.stack(probs[test])
        indiv = {n: summary(y_true, q.argmax(axis=1), class_names) for n, q in zip(args.nomes, p)}
        accs = [indiv[n]["acuracia"] for n in args.nomes]
        results[test] = {"individual": indiv, "media": float(np.mean(accs)),
                         "desvio": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                         "ensemble": summary(y_true, p.mean(axis=0).argmax(axis=1), class_names)}

    out_dir.mkdir(parents=True)
    save_json({"modelos": dict(zip(args.nomes, args.modelos)), "pasta": args.pasta, "classes": class_names,
               "regra_ensemble": "média simples das probabilidades (soft voting)", "resultados": results},
              out_dir / "resultados_oclusao_ensemble.json")

    base_mean, base_ens = results["original"]["media"], results["original"]["ensemble"]["acuracia"]
    print(f"\nOclusão e cor no TESTE ({len(y_true)} imagens)")
    print(f"{'teste':<18}" + "".join(f"{n:>8}" for n in args.nomes)
          + f"{'média ± DP':>16}{'queda':>8}{'ENSEMBLE':>10}{'queda':>8}")
    for test, label in TESTS.items():
        r = results[test]
        print(f"{label:<18}" + "".join(f"{r['individual'][n]['acuracia'] * 100:>7.1f}%" for n in args.nomes)
              + f"{r['media'] * 100:>9.1f} ± {r['desvio'] * 100:<4.1f}{(r['media'] - base_mean) * 100:>8.1f}"
              + f"{r['ensemble']['acuracia'] * 100:>9.1f}%{(r['ensemble']['acuracia'] - base_ens) * 100:>8.1f}")
    print("\nEnsemble, acerto por moeda:")
    print(f"{'teste':<18}" + "".join(f"{short(c):>12}" for c in class_names))
    for test, label in TESTS.items():
        rec = results[test]["ensemble"]["acuracia_por_classe"]
        print(f"{label:<18}" + "".join(f"{rec[c] * 100:>11.1f}%" for c in class_names))
    print(f"\nResultados salvos em: {out_dir}")


if __name__ == "__main__":
    main()
