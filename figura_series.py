import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import ARTIFACTS_DIR, BASE_DIR, DATASET_DIR, SEED
from make_group_split import FACE_WORDS, series_key

# Figura para o texto: algumas fotos de cada série escolhida, uma série por linha. Mostra que as fotos de uma
# mesma série têm o mesmo fundo, luz e enquadramento. Na coleção "dois números" da Washington, cada foto vem de
# um par de números diferente, para mostrar que todos os pares têm o mesmo estilo. Só lê o dataset/raw.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
DEFAULT_SERIES = [
    "Jefferson Nickel 30916",
    "Lincoln Cent 41463",
    "Washington Quarter 17345",
    "Washington Quarter (coleção dois números)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figura com fotos de algumas séries do U.S. Coins.")
    parser.add_argument("--series", nargs="+", default=DEFAULT_SERIES, help="Séries (uma por linha da figura)")
    parser.add_argument("--por_serie", type=int, default=4, help="Fotos por série")
    parser.add_argument("--teste", default=str(BASE_DIR / "dataset" / "split_series_v2" / "teste"))
    parser.add_argument("--saida", default=str(ARTIFACTS_DIR / "experiments" / "figura_series"))
    return parser.parse_args()


def old_key(path: Path) -> str:
    # Chave da primeira versão da divisão: o nome sem o último "_" (ex.: "Washington Quarter 11 6")
    stem = path.stem
    for word in FACE_WORDS:
        if stem.lower().endswith(word):
            stem = stem[: -len(word)]
    return stem.rsplit("_", 1)[0].strip() if "_" in stem else stem.strip()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.saida)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for class_dir in sorted(p for p in DATASET_DIR.iterdir() if p.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                groups[series_key(path)].append(path)
    test_names = {p.name for p in Path(args.teste).rglob("*") if p.is_file()} if Path(args.teste).exists() else set()
    missing = [s for s in args.series if s not in groups]
    if missing:
        raise SystemExit("Séries não encontradas: " + ", ".join(missing))

    rng = np.random.default_rng(SEED)
    n = args.por_serie
    fig, axes = plt.subplots(len(args.series), n, figsize=(2.2 * n, 2.5 * len(args.series)), squeeze=False)
    for row, name in enumerate(args.series):
        files = groups[name]
        by_old = defaultdict(list)
        for p in files:
            by_old[old_key(p)].append(p)
        if len(by_old) > 1:  # coleção "dois números": uma foto de cada par diferente
            keys = rng.choice(sorted(by_old), size=min(n, len(by_old)), replace=False)
            chosen = [by_old[k][0] for k in keys]
        else:
            chosen = list(rng.choice(files, size=min(n, len(files)), replace=False))
        part = "teste" if files[0].name in test_names else "treino"
        print(f"{name}: {len(files)} fotos ({part})" + (f", {len(by_old)} pares de números" if len(by_old) > 1 else ""))
        for col in range(n):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(chosen):
                ax.imshow(Image.open(chosen[col]).convert("RGB"))
                if len(by_old) > 1:
                    ax.set_title(old_key(chosen[col]).replace("Washington Quarter ", "nº "), fontsize=8)
        label = name.replace(" (coleção dois números)", "\n(coleção dois números)")
        axes[row, 0].text(-0.08, 0.5, f"{label}\n[{part}]", transform=axes[row, 0].transAxes,
                          ha="right", va="center", fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(left=0.22)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"figura_series.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva em: {out_dir} (figura_series.png e figura_series.pdf)")


if __name__ == "__main__":
    main()
