import argparse
import csv
import glob
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import ARTIFACTS_DIR, BASE_DIR, BATCH_SIZE, DATASET_DIR, IMAGE_SIZE
from io_utils import save_json
from make_group_split import series_key

# Mostra, para cada SÉRIE do teste, quantas fotos cada modelo acertou e para onde foram os erros.
# Também separa as fotos de Washington que viram Lincoln e compara o brilho e o tom (vermelho - azul) delas com
# as fotos de Washington acertadas. Não treina nada; só usa modelos já salvos.
MODEL_FILE = "modelo_divisao_series.keras"
COIN_RADIUS = 0.45  # mesmo círculo do teste de oclusão (tom e brilho medidos só dentro da moeda)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Erros de vários modelos por série do teste.")
    parser.add_argument("--padrao", nargs="+", required=True,
                        help='Pastas de resultados (aceita *), ex.: "artifacts/experiments/divisao_series_v2_profundo*"')
    parser.add_argument("--pasta", default=str(BASE_DIR / "dataset" / "split_series_v2" / "teste"))
    parser.add_argument("--saida", default=str(ARTIFACTS_DIR / "experiments" / "erros_por_serie"))
    return parser.parse_args()


def find_models(patterns: list) -> dict:
    models = {}
    for pattern in patterns:
        for folder in sorted(glob.glob(pattern)):
            path = Path(folder) / MODEL_FILE
            if path.exists():
                models[Path(folder).name.replace("divisao_series_v2_", "").replace("divisao_series_", "")] = path
    return models


def coin_stats(image: np.ndarray) -> tuple:
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    inside = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 <= (COIN_RADIUS * min(h, w)) ** 2
    pixels = image[inside]
    return float(pixels.mean()), float(pixels[:, 0].mean() - pixels[:, 2].mean())  # brilho, tom (R - B)


def short(name: str) -> str:
    return name.split()[0][0]  # J / L / W


def save_grid(items: list, title: str, path: Path) -> None:
    if not items:
        return
    items = items[:30]
    cols = 6
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.7 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (img, label) in zip(axes.flat, items):
        ax.imshow(img)
        ax.set_title(label, fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.saida)
    if out_dir.exists():
        raise SystemExit(f"A pasta {out_dir} já existe. Use outra --saida.")
    models = find_models(args.padrao)
    if not models:
        raise SystemExit("Nenhum modelo encontrado. Confira o --padrao.")
    print("Modelos:", ", ".join(models))

    class_names = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())
    test_ds = tf.keras.utils.image_dataset_from_directory(
        args.pasta, labels="inferred", label_mode="categorical", class_names=class_names,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False)
    files = [Path(p) for p in test_ds.file_paths]
    images = np.concatenate([x.numpy() for x, _ in test_ds]).astype("uint8")
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds])
    series = [series_key(p) for p in files]

    preds, confs = {}, {}
    for name, path in models.items():
        probs = tf.keras.models.load_model(path).predict(test_ds, verbose=0)
        preds[name], confs[name] = probs.argmax(axis=1), probs.max(axis=1)
        print(f"  {name:<40} teste {np.mean(preds[name] == y_true) * 100:5.1f}%")
    names = list(models)
    wrong_count = sum((preds[n] != y_true).astype(int) for n in names)

    # 1) Acerto e destino dos erros por série
    lines = []
    summary = {}
    for s in sorted(set(series), key=lambda k: (class_names[y_true[series.index(k)]], k)):
        idx = [i for i, k in enumerate(series) if k == s]
        cls = class_names[y_true[idx[0]]]
        summary[s] = {"classe": cls, "imagens": len(idx), "modelos": {}}
        lines.append(f"\n{s}  ({cls.split(',')[0]}, {len(idx)} fotos)")
        for n in names:
            dest = Counter(short(class_names[preds[n][i]]) for i in idx)
            hits = sum(preds[n][i] == y_true[i] for i in idx)
            summary[s]["modelos"][n] = {"acertos": int(hits), "previsoes": dict(dest)}
            lines.append(f"  {n:<40} acerto {hits / len(idx) * 100:5.1f}%   previsões J/L/W = "
                         f"{dest.get('J', 0):>3} / {dest.get('L', 0):>3} / {dest.get('W', 0):>3}")
    print("\nACERTO POR SÉRIE")
    print("\n".join(lines))

    # 2) Washington -> Lincoln: tom e brilho das fotos erradas x acertadas (voto da maioria dos modelos)
    w_idx = class_names.index(next(c for c in class_names if c.startswith("Washington")))
    l_idx = class_names.index(next(c for c in class_names if c.startswith("Lincoln")))
    votes = np.stack([preds[n] for n in names])
    majority = np.array([np.bincount(v, minlength=len(class_names)).argmax() for v in votes.T])
    groups = {"Washington acertado (maioria)": [], "Washington -> Lincoln (maioria)": []}
    stats = {}
    for i in range(len(files)):
        if y_true[i] != w_idx:
            continue
        if majority[i] == w_idx:
            groups["Washington acertado (maioria)"].append(i)
        elif majority[i] == l_idx:
            groups["Washington -> Lincoln (maioria)"].append(i)
    lincoln_true = [i for i in range(len(files)) if y_true[i] == l_idx]
    print("\nBRILHO E TOM DENTRO DA MOEDA (0-255; tom = vermelho - azul, quanto maior mais quente)")
    for label, idx in list(groups.items()) + [("Lincoln (todos, para comparar)", lincoln_true)]:
        if not idx:
            continue
        values = np.array([coin_stats(images[i]) for i in idx])
        per_series = Counter(series[i] for i in idx)
        stats[label] = {"fotos": len(idx), "brilho_medio": float(values[:, 0].mean()),
                        "tom_medio": float(values[:, 1].mean()), "por_serie": dict(per_series)}
        print(f"  {label:<36} {len(idx):>4} fotos   brilho {values[:, 0].mean():6.1f}   tom {values[:, 1].mean():6.1f}"
              f"   séries: {dict(per_series)}")

    # 3) Arquivos de saída
    out_dir.mkdir(parents=True)
    with open(out_dir / "por_imagem.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["arquivo", "classe", "serie", "modelos_que_erraram", "brilho", "tom"]
                        + [f"{n}_previsao" for n in names] + [f"{n}_confianca" for n in names])
        for i, p in enumerate(files):
            b, t = coin_stats(images[i])
            writer.writerow([p.name, class_names[y_true[i]], series[i], int(wrong_count[i]), f"{b:.1f}", f"{t:.1f}"]
                            + [short(class_names[preds[n][i]]) for n in names] + [f"{confs[n][i]:.2f}" for n in names])
    save_json({"modelos": {n: str(p) for n, p in models.items()}, "por_serie": summary, "washington_tom_brilho": stats},
              out_dir / "resumo.json")
    always_wrong = [i for i in groups["Washington -> Lincoln (maioria)"]]
    always_right = groups["Washington acertado (maioria)"]
    save_grid([(images[i], f"{series[i].split()[-1]} | erraram {wrong_count[i]}/{len(names)}") for i in always_wrong],
              "Washington que a maioria dos modelos chama de LINCOLN", out_dir / "washington_vira_lincoln.png")
    save_grid([(images[i], f"{series[i].split()[-1]} | erraram {wrong_count[i]}/{len(names)}") for i in always_right],
              "Washington que a maioria dos modelos ACERTA", out_dir / "washington_acertado.png")
    print(f"\nArquivos salvos em: {out_dir}")
    print("  por_imagem.csv (uma linha por foto), resumo.json, washington_vira_lincoln.png, washington_acertado.png")


if __name__ == "__main__":
    main()
