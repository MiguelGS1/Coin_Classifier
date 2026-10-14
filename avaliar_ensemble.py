import argparse
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import DATASET_DIR
from dataset_loader import build_single_image_tensor
from io_utils import save_json
from metrics import build_confusion_matrix, compute_metrics, plot_confusion_matrix

# Ensemble: cada foto passa por todos os modelos (ex.: as 3 sementes do modelo final) e a resposta é a classe com a
# MAIOR MÉDIA das probabilidades. Média simples, sem pesos e sem escolher modelos. Mostra também o resultado de cada
# modelo sozinho, na mesma pasta, para comparar. Funciona com qualquer pasta organizada por classe (teste, World Coins,
# celular). Não treina nada e não altera a pasta.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia o ensemble (média das probabilidades) de vários modelos.")
    parser.add_argument("--modelos", nargs="+", required=True, help="Arquivos .keras (ex.: as 3 sementes)")
    parser.add_argument("--nomes", nargs="+", required=True, help="Um nome curto por modelo (ex.: s42 s1 s2)")
    parser.add_argument("--pasta", required=True, help="Pasta com uma subpasta por classe")
    parser.add_argument("--saida", required=True, help="Pasta onde salvar os resultados")
    return parser.parse_args()


def short(name: str) -> str:
    return name.split()[0]


def main() -> None:
    args = parse_args()
    if len(args.modelos) != len(args.nomes):
        raise SystemExit("Passe um nome para cada modelo (--nomes).")
    out_dir = Path(args.saida)
    if out_dir.exists():
        raise SystemExit(f"A pasta {out_dir} já existe. Use outra --saida.")
    class_names = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())
    folder = Path(args.pasta)

    paths, labels = [], []
    for class_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        if class_dir.name not in class_names:
            raise SystemExit(f"Pasta '{class_dir.name}' não é uma das classes do modelo: {class_names}")
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(p)
                labels.append(class_names.index(class_dir.name))
    if not paths:
        raise SystemExit(f"Nenhuma foto encontrada nas subpastas de {folder}")
    labels = np.array(labels)
    print(f"Fotos: {len(paths)}", flush=True)

    probs = []  # [modelo, foto, classe]
    for model_path, nome in zip(args.modelos, args.nomes):
        print(f"Avaliando {nome} ({model_path}) ...", flush=True)
        model = tf.keras.models.load_model(model_path)
        p = []
        for start in range(0, len(paths), 32):
            batch = tf.concat([build_single_image_tensor(str(x)) for x in paths[start:start + 32]], axis=0)
            p.append(model(batch, training=False).numpy())
        probs.append(np.concatenate(p))
    probs = np.stack(probs)
    mean_probs = probs.mean(axis=0)

    out_dir.mkdir(parents=True)
    results = {"modelos": dict(zip(args.nomes, args.modelos)), "pasta": str(folder), "individual": {}}
    print(f"\n{'Modelo':<12}{'Acurácia':>10}{'Acur. balanceada':>18}")
    accs = []
    for nome, p in zip(args.nomes, probs):
        m = compute_metrics(build_confusion_matrix(labels, p.argmax(axis=1), len(class_names)), class_names)
        results["individual"][nome] = m
        accs.append(m["acuracia"])
        print(f"{nome:<12}{m['acuracia'] * 100:>9.1f}%{m['revocacao_macro'] * 100:>17.1f}%")
    sd = np.std(accs, ddof=1) * 100 if len(accs) > 1 else 0.0
    print(f"{'média':<12}{np.mean(accs) * 100:>9.1f}%  (± {sd:.1f})")

    matrix = build_confusion_matrix(labels, mean_probs.argmax(axis=1), len(class_names))
    ens = compute_metrics(matrix, class_names)
    results["media_individual"] = {"acuracia": float(np.mean(accs)), "desvio": float(sd / 100)}
    results["ensemble"] = ens
    print(f"{'ENSEMBLE':<12}{ens['acuracia'] * 100:>9.1f}%{ens['revocacao_macro'] * 100:>17.1f}%")
    print("Matriz do ensemble (linhas = moeda certa, colunas = resposta):")
    for name, row in zip(class_names, matrix):
        print(f"  {short(name):<12}{row}")

    save_json(results, out_dir / "resultados_ensemble.json")
    plot_confusion_matrix(matrix, class_names, out_dir / "matriz_confusao_ensemble.png")
    with open(out_dir / "fotos_ensemble.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["arquivo", "moeda_certa", "ensemble", "acertou"]
                        + [f"{n}_resposta" for n in args.nomes]
                        + [f"ens_prob_{short(c)}" for c in class_names])
        for i, path in enumerate(paths):
            pred = int(mean_probs[i].argmax())
            writer.writerow([path.name, short(class_names[labels[i]]), short(class_names[pred]),
                             "sim" if pred == labels[i] else "NAO"]
                            + [short(class_names[int(p[i].argmax())]) for p in probs]
                            + [f"{x * 100:.1f}" for x in mean_probs[i]])
    print(f"\nSalvo em: {out_dir}")
    print("  resultados_ensemble.json, matriz_confusao_ensemble.png, fotos_ensemble.csv (foto por foto)")


if __name__ == "__main__":
    main()
