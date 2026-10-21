import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

from config import ARTIFACTS_DIR, BASE_DIR, SEED
from make_group_split import series_key

# Figura para o texto: para cada moeda (uma por linha), algumas fotos do TREINO e algumas do TESTE da divisão v2.
# Cada foto vem de uma série diferente (o nome da série aparece embaixo). As fotos são sorteadas (--semente) e
# completadas com branco até ficarem quadradas, para todas terem o mesmo tamanho. Só lê as pastas; não altera nada.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
TRAIN_COLOR = "#1f4e79"  # azul escuro
TEST_COLOR = "#c55a11"   # laranja escuro
NAMES = {"Jefferson": "Jefferson Nickel", "Lincoln": "Lincoln Cent", "Washington": "Washington Quarter"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figura com fotos do treino e do teste de cada moeda.")
    parser.add_argument("--pasta", default=str(BASE_DIR / "dataset" / "split_series_v2"))
    parser.add_argument("--por_lado", type=int, default=2, help="Fotos do treino e do teste em cada linha")
    parser.add_argument("--semente", type=int, default=SEED, help="Troque para sortear outras fotos")
    parser.add_argument("--saida", default=str(ARTIFACTS_DIR / "experiments" / "figura_treino_teste"))
    return parser.parse_args()


def pick(folder: Path, n: int, rng: np.random.Generator) -> list:
    # Uma foto de cada série (sorteando as séries); só repete série se houver menos séries que n
    groups = defaultdict(list)
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            groups[series_key(p)].append(p)
    keys = list(rng.permutation(sorted(groups)))
    chosen = []
    while len(chosen) < n:
        added = False
        for k in keys:
            remaining = [p for p in groups[k] if p not in chosen]
            if remaining and len(chosen) < n:
                chosen.append(remaining[int(rng.integers(len(remaining)))])
                added = True
        if not added:
            break
    return chosen


def to_square(path: Path, size: int = 600) -> Image.Image:
    # Reduz mantendo a proporção e completa com branco até ficar quadrada (não corta a moeda)
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def short_series(path: Path) -> str:
    key = series_key(path)
    return "coleção dois números" if "dois números" in key else "série " + key.split()[-1]


def main() -> None:
    args = parse_args()
    root, out_dir = Path(args.pasta), Path(args.saida)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.semente)
    classes = sorted(p.name for p in (root / "teste").iterdir() if p.is_dir())
    n = args.por_lado

    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    # Colunas: n do treino, um espaço com a linha divisória, n do teste
    width_ratios = [1] * n + [0.18] + [1] * n
    fig = plt.figure(figsize=(1.55 * 2 * n + 1.3, 1.85 * len(classes) + 0.45))
    grid = GridSpec(len(classes), 2 * n + 1, figure=fig, width_ratios=width_ratios,
                    wspace=0.08, hspace=0.32, left=0.13, right=0.995, top=0.90, bottom=0.03)

    for row, cls in enumerate(classes):
        train = pick(root / "treino_validacao" / cls, n, rng)
        test = pick(root / "teste" / cls, n, rng)
        print(f"{cls}: treino = {[p.name for p in train]} | teste = {[p.name for p in test]}")
        cells = [(i, p, TRAIN_COLOR) for i, p in enumerate(train)] + \
                [(n + 1 + i, p, TEST_COLOR) for i, p in enumerate(test)]
        for col, path, color in cells:
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(to_square(path))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.6)
            ax.set_xlabel(short_series(path), fontsize=8, labelpad=2)
            if col == 0:
                ax.set_ylabel(NAMES.get(cls.split()[0], cls.split(",")[0]), fontsize=9.5,
                              fontstyle="italic", labelpad=6)
        sep = fig.add_subplot(grid[row, n])
        sep.axis("off")
        sep.axvline(0.5, color="0.6", linestyle="--", linewidth=0.8)
        sep.set_xlim(0, 1)

    # Títulos sobre cada lado, centralizados nas colunas
    left_mid = fig.add_subplot(grid[0, :n]).get_position()
    right_mid = fig.add_subplot(grid[0, n + 1:]).get_position()
    for ax in fig.axes[-2:]:
        ax.remove()
    fig.text((left_mid.x0 + left_mid.x1) / 2, 0.955, "Treino (séries vistas)", ha="center", va="center",
             fontsize=10.5, fontweight="bold", color=TRAIN_COLOR)
    fig.text((right_mid.x0 + right_mid.x1) / 2, 0.955, "Teste (séries nunca vistas)", ha="center", va="center",
             fontsize=10.5, fontweight="bold", color=TEST_COLOR)

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"figura_treino_teste.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Figura salva em: {out_dir} (figura_treino_teste.png e figura_treino_teste.pdf)")


if __name__ == "__main__":
    main()
