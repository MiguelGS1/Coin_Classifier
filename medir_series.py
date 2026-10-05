import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import ARTIFACTS_DIR, BASE_DIR, IMAGE_SIZE
from io_utils import save_json
from make_group_split import series_key

# Mede o brilho e o tom (vermelho - azul) de cada SÉRIE, dentro da moeda e no fundo, para o treino e o teste da
# divisão v2. Serve para ver se alguma série do TREINO tem fotos parecidas com as de uma série do teste (ex.: a
# luz rosada/escura da Washington 17345). Não usa nenhum modelo e não treina nada; só lê as imagens.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
COIN_RADIUS = 0.45  # mesmo círculo do teste de oclusão


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brilho e tom de cada série (treino e teste).")
    parser.add_argument("--pasta", default=str(BASE_DIR / "dataset" / "split_series_v2"))
    parser.add_argument("--saida", default=str(ARTIFACTS_DIR / "experiments" / "brilho_tom_series"))
    return parser.parse_args()


def masks() -> tuple:
    h, w = IMAGE_SIZE
    yy, xx = np.mgrid[0:h, 0:w]
    coin = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 <= (COIN_RADIUS * min(h, w)) ** 2
    return coin, ~coin


def measure(path: Path, coin: np.ndarray, background: np.ndarray) -> list:
    image = np.asarray(Image.open(path).convert("RGB").resize(IMAGE_SIZE[::-1]), dtype=np.float32)
    values = []
    for region in (coin, background):
        pixels = image[region]
        values += [float(pixels.mean()), float(pixels[:, 0].mean() - pixels[:, 2].mean())]
    return values  # brilho_moeda, tom_moeda, brilho_fundo, tom_fundo


def main() -> None:
    args = parse_args()
    root, out_dir = Path(args.pasta), Path(args.saida)
    if out_dir.exists():
        raise SystemExit(f"A pasta {out_dir} já existe. Use outra --saida.")
    coin, background = masks()

    rows = []
    for part in ("treino_validacao", "teste"):
        for class_dir in sorted(p for p in (root / part).iterdir() if p.is_dir()):
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append([part, class_dir.name, series_key(path), path.name] + measure(path, coin, background))
    print(f"Imagens medidas: {len(rows)}")

    groups = defaultdict(list)
    for row in rows:
        groups[(row[1], row[0], row[2])].append(row[4:])
    summary = []
    for (cls, part, s), values in groups.items():
        v = np.array(values)
        summary.append({"classe": cls, "parte": part, "serie": s, "fotos": len(v),
                        "brilho_moeda": float(v[:, 0].mean()), "tom_moeda": float(v[:, 1].mean()),
                        "brilho_fundo": float(v[:, 2].mean()), "tom_fundo": float(v[:, 3].mean())})
    summary.sort(key=lambda r: (r["classe"], r["parte"] != "teste", -r["fotos"]))

    print("\nPor série (médias; brilho de 0 a 255; tom = vermelho - azul, maior = mais quente/acobreado)")
    print(f"{'série':<46}{'parte':>8}{'fotos':>7}{'brilho moeda':>14}{'tom moeda':>11}{'brilho fundo':>14}{'tom fundo':>11}")
    current = None
    for r in summary:
        if r["classe"] != current:
            current = r["classe"]
            print(f"\n{current}")
        part = "TESTE" if r["parte"] == "teste" else "treino"
        print(f"  {r['serie'][:44]:<44}{part:>8}{r['fotos']:>7}{r['brilho_moeda']:>14.1f}{r['tom_moeda']:>11.1f}"
              f"{r['brilho_fundo']:>14.1f}{r['tom_fundo']:>11.1f}")

    out_dir.mkdir(parents=True)
    with open(out_dir / "por_imagem.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["parte", "classe", "serie", "arquivo", "brilho_moeda", "tom_moeda", "brilho_fundo", "tom_fundo"])
        writer.writerows([r[:4] + [f"{x:.1f}" for x in r[4:]] for r in rows])
    save_json(summary, out_dir / "por_serie.json")

    # Gráfico: cada ponto é uma série (tamanho = número de fotos); teste com borda preta e nome
    colors = {"Jefferson": "tab:blue", "Lincoln": "tab:orange", "Washington": "tab:green"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (bx, tx, title) in zip(axes, [("brilho_moeda", "tom_moeda", "Dentro da moeda"),
                                          ("brilho_fundo", "tom_fundo", "Fundo (fora do círculo)")]):
        for r in summary:
            color = colors[r["classe"].split()[0]]
            test = r["parte"] == "teste"
            ax.scatter(r[bx], r[tx], s=20 + r["fotos"] / 2, color=color, alpha=0.6,
                       edgecolors="black" if test else "none", linewidths=1.5 if test else 0)
            if test:
                ax.annotate(r["serie"].split()[-1], (r[bx], r[tx]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Brilho (0-255)")
        ax.set_ylabel("Tom (vermelho - azul)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    for name, color in colors.items():
        axes[0].scatter([], [], color=color, label=name)
    axes[0].scatter([], [], facecolors="none", edgecolors="black", label="série do TESTE")
    axes[0].legend(fontsize=8)
    fig.suptitle("Cada ponto é uma série (tamanho = número de fotos)")
    fig.tight_layout()
    fig.savefig(out_dir / "brilho_tom_series.png", dpi=130)
    plt.close(fig)
    print(f"\nArquivos salvos em: {out_dir}  (por_serie.json, por_imagem.csv, brilho_tom_series.png)")


if __name__ == "__main__":
    main()
