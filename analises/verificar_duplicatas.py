import argparse
import csv
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import ARTIFACTS_DIR, BASE_DIR

# Procura fotos REPETIDAS ou QUASE IGUAIS (mesma foto redimensionada, recomprimida, com outra cor) entre o
# treino+validação e o teste da divisão por série, e também dentro do próprio teste.
#   - cópia exata: mesmo conteúdo do arquivo (MD5);
#   - quase igual: "impressão digital" da imagem (dHash de 64 bits) com até --limiar bits diferentes.
# O dHash não pega a mesma moeda fotografada de novo (outra foto), só versões da MESMA foto. Os pares encontrados
# precisam ser olhados (figura pares_*.png): pode haver falsos alarmes. Só lê as pastas; não altera nada.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Procura fotos repetidas ou quase iguais entre treino e teste.")
    parser.add_argument("--pasta", default=str(BASE_DIR / "dataset" / "split_series_v2"))
    parser.add_argument("--limiar", type=int, default=6, help="Máximo de bits diferentes para 'quase igual' (0-64)")
    parser.add_argument("--saida", default=str(ARTIFACTS_DIR / "experiments" / "verificacao_duplicatas_v2"))
    return parser.parse_args()


def dhash(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((9, 8), Image.LANCZOS)
    pixels = np.asarray(image, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def load(folder: Path) -> list:
    items = []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            items.append({"path": p, "classe": p.parent.name.split()[0],
                          "md5": hashlib.md5(p.read_bytes()).hexdigest(), "hash": dhash(p)})
    return items


def near_pairs(a: list, b: list, limit: int, same: bool) -> list:
    ha = np.array([x["hash"] for x in a])
    hb = np.array([x["hash"] for x in b])
    pairs = []
    for i in range(len(a)):
        dist = (ha[i][None, :] != hb).sum(axis=1)
        for j in np.flatnonzero(dist <= limit):
            if same and j <= i:
                continue
            pairs.append((a[i], b[j], int(dist[j])))
    return sorted(pairs, key=lambda x: x[2])


def save_figure(pairs: list, path: Path, title: str) -> None:
    pairs = pairs[:40]
    if not pairs:
        return
    fig, axes = plt.subplots(len(pairs), 2, figsize=(5, 2.4 * len(pairs)), squeeze=False)
    for (x, y, d), row in zip(pairs, axes):
        for ax, item in zip(row, (x, y)):
            ax.imshow(Image.open(item["path"]).convert("RGB"))
            ax.set_title(item["path"].name[:34], fontsize=6)
            ax.axis("off")
        row[0].text(-0.05, 0.5, f"dist {d}", transform=row[0].transAxes, ha="right", fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root, out_dir = Path(args.pasta), Path(args.saida)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Lendo as fotos (pode levar alguns minutos)...", flush=True)
    train, test = load(root / "treino_validacao"), load(root / "teste")
    print(f"treino+validação: {len(train)}   teste: {len(test)}")

    train_md5 = {x["md5"] for x in train}
    exact_cross = [x for x in test if x["md5"] in train_md5]
    seen, exact_test = {}, []
    for x in test:
        if x["md5"] in seen:
            exact_test.append((seen[x["md5"]], x, 0))
        else:
            seen[x["md5"]] = x
    cross = near_pairs(test, train, args.limiar, same=False)
    inside = near_pairs(test, test, args.limiar, same=True)

    print(f"\nCópias EXATAS teste x treino: {len(exact_cross)}")
    print(f"Cópias EXATAS dentro do teste: {len(exact_test)} pares")
    print(f"QUASE IGUAIS teste x treino (até {args.limiar} bits): {len(cross)} pares")
    print(f"QUASE IGUAIS dentro do teste (até {args.limiar} bits): {len(inside)} pares")
    for name, pairs in (("teste x treino", cross), ("dentro do teste", inside)):
        for x, y, d in pairs[:15]:
            other = "  << CLASSES DIFERENTES" if x["classe"] != y["classe"] else ""
            print(f"  [{name}] dist {d:>2}: {x['path'].name}  <->  {y['path'].name}{other}")

    with open(out_dir / "pares.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["tipo", "distancia", "foto_teste", "outra_foto", "classe_1", "classe_2"])
        for name, pairs in (("exata_dentro_teste", exact_test), ("quase_teste_x_treino", cross),
                            ("quase_dentro_teste", inside)):
            for x, y, d in pairs:
                writer.writerow([name, d, x["path"].name, y["path"].name, x["classe"], y["classe"]])
    save_figure(cross, out_dir / "pares_teste_x_treino.png", "Quase iguais: teste (esq.) x treino (dir.)")
    save_figure(inside, out_dir / "pares_dentro_teste.png", "Quase iguais dentro do teste")
    print(f"\nSalvo em: {out_dir} (pares.csv e figuras pares_*.png, quando houver pares)")


if __name__ == "__main__":
    main()
