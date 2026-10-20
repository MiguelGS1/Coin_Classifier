import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from prever_moeda_recortada import isolate_coin, load_rgb

# Figura para o texto: para cada foto escolhida, a foto original (em cima) e a imagem recortada que a rede recebe na
# configuração com recorte (embaixo). Se os CSV das avaliações forem passados, escreve embaixo de cada coluna a moeda
# certa e a resposta das duas configurações (serve para a figura de exemplos de ERROS). Só lê as fotos; não classifica
# nada e não altera nenhuma pasta. Use só fotos do U.S. Coins (as do World Coins não podem ser redistribuídas).
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figura com fotos originais e recortadas.")
    parser.add_argument("--pasta", required=True, help="Pasta com uma subpasta por classe (ex.: o teste v2)")
    parser.add_argument("--fotos", nargs="+", required=True, help="Nomes dos arquivos (ex.: \"Jefferson Nickel 30916_49.jpg\")")
    parser.add_argument("--csv_a", default=None, help="fotos_ensemble.csv (sem recorte), para mostrar a resposta")
    parser.add_argument("--csv_b", default=None, help="fotos_ensemble_recorte.csv (com recorte), para mostrar a resposta")
    parser.add_argument("--saida", required=True, help="Arquivo .png da figura")
    parser.add_argument("--tamanho", type=float, default=2.2, help="Largura de cada coluna, em polegadas")
    return parser.parse_args()


def read_answers(path):
    if not path:
        return {}
    with open(path, encoding="utf-8-sig") as fh:
        return {r["arquivo"]: r for r in csv.DictReader(fh, delimiter=";")}


def main() -> None:
    args = parse_args()
    files = {p.name: p for p in Path(args.pasta).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS}
    missing = [f for f in args.fotos if f not in files]
    if missing:
        raise SystemExit("Fotos não encontradas na pasta:\n  " + "\n  ".join(missing))
    ans_a, ans_b = read_answers(args.csv_a), read_answers(args.csv_b)

    n = len(args.fotos)
    fig, axes = plt.subplots(2, n, figsize=(args.tamanho * n, args.tamanho * 2 + 0.6), squeeze=False)
    for j, name in enumerate(args.fotos):
        rgb = load_rgb(str(files[name]))
        coin, found = isolate_coin(rgb, 255)
        axes[0, j].imshow(rgb)
        axes[1, j].imshow(np.asarray(Image.fromarray(coin).resize((224, 224), Image.BILINEAR)))
        for ax in axes[:, j]:
            ax.set_xticks([])
            ax.set_yticks([])
        label = files[name].parent.name.split()[0]
        lines = [f"({chr(97 + j)}) {label}"]
        if name in ans_a:
            lines.append(f"sem recorte: {ans_a[name]['ensemble']}")
        if name in ans_b:
            lines.append(f"com recorte: {ans_b[name]['ensemble']}")
        if not found:
            lines.append("(círculo não encontrado)")
        axes[1, j].set_xlabel("\n".join(lines), fontsize=9)
    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Recortada", fontsize=10)
    fig.tight_layout()
    out = Path(args.saida)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Figura salva em: {out}")
    for j, name in enumerate(args.fotos):
        print(f"  ({chr(97 + j)}) {name}")


if __name__ == "__main__":
    main()
