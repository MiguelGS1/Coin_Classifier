import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import ARTIFACTS_DIR, BASE_DIR

# Segunda etapa da verificação de duplicatas. O verificar_duplicatas.py usa uma "impressão digital" pequena (8x8),
# que dá muitos falsos alarmes com moedas (toda foto é "um disco claro num fundo"). Aqui, cada par suspeito do
# pares.csv é comparado com mais detalhe:
#   - semelhança de pixels: correlação entre as duas fotos em tons de cinza, 128x128 (1,0 = idênticas);
#   - impressão digital maior (dHash 16x16 = 256 bits; 0 = iguais).
# Os pares mais parecidos vão para uma figura, para conferir a olho. Só lê; não altera nada.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Confere com mais detalhe os pares suspeitos de duplicata.")
    parser.add_argument("--pasta", default=str(BASE_DIR / "dataset" / "split_series_v2"))
    parser.add_argument("--entrada", default=str(ARTIFACTS_DIR / "experiments" / "verificacao_duplicatas_v2"))
    parser.add_argument("--figura", type=int, default=30, help="Quantos pares (os mais parecidos) mostrar na figura")
    return parser.parse_args()


def gray(path: Path, size: int) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32)


def correlation(a: Path, b: Path) -> float:
    x, y = gray(a, 128).flatten(), gray(b, 128).flatten()
    x, y = x - x.mean(), y - y.mean()
    return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))


def dhash256(path: Path) -> np.ndarray:
    pixels = gray(path, 17)[:16, :]
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def main() -> None:
    args = parse_args()
    root, in_dir = Path(args.pasta), Path(args.entrada)
    where = {}
    for part in ("treino_validacao", "teste"):
        for p in (root / part).rglob("*"):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                where[(part, p.name)] = p

    with open(in_dir / "pares.csv", encoding="utf-8-sig") as fh:
        pairs = list(csv.DictReader(fh, delimiter=";"))
    rows = []
    for r in pairs:
        other_part = "treino_validacao" if r["tipo"] == "quase_teste_x_treino" else "teste"
        a, b = where.get(("teste", r["foto_teste"])), where.get((other_part, r["outra_foto"]))
        if a is None or b is None:
            continue
        rows.append({**r, "a": a, "b": b, "corr": correlation(a, b),
                     "dhash256": int((dhash256(a) != dhash256(b)).sum())})
    rows.sort(key=lambda r: -r["corr"])

    print(f"Pares conferidos: {len(rows)}")
    for tipo in ("exata_dentro_teste", "quase_teste_x_treino", "quase_dentro_teste"):
        sel = [r for r in rows if r["tipo"] == tipo]
        if not sel:
            continue
        c = np.array([r["corr"] for r in sel])
        print(f"\n{tipo}: {len(sel)} pares | correlação >= 0,98: {(c >= 0.98).sum()} | 0,95-0,98: "
              f"{((c >= 0.95) & (c < 0.98)).sum()} | 0,90-0,95: {((c >= 0.90) & (c < 0.95)).sum()} | < 0,90: {(c < 0.90).sum()}")
        for r in sel[:12]:
            other = "  << CLASSES DIFERENTES" if r["classe_1"] != r["classe_2"] else ""
            print(f"  corr {r['corr']:.3f}  dhash256 {r['dhash256']:>3}  {r['foto_teste']}  <->  {r['outra_foto']}{other}")

    out = in_dir / "pares_refinados.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["tipo", "correlacao", "dhash256", "dhash64", "foto_teste", "outra_foto", "classe_1", "classe_2"])
        for r in rows:
            writer.writerow([r["tipo"], f"{r['corr']:.4f}", r["dhash256"], r["distancia"], r["foto_teste"],
                             r["outra_foto"], r["classe_1"], r["classe_2"]])

    for tipo, name in (("quase_teste_x_treino", "refinado_teste_x_treino.png"),
                       ("quase_dentro_teste", "refinado_dentro_teste.png")):
        sel = [r for r in rows if r["tipo"] == tipo][: args.figura]
        if not sel:
            continue
        fig, axes = plt.subplots(len(sel), 2, figsize=(5.4, 2.5 * len(sel)), squeeze=False)
        for r, row in zip(sel, axes):
            for ax, path in zip(row, (r["a"], r["b"])):
                ax.imshow(Image.open(path).convert("RGB"))
                ax.set_title(path.name[:36], fontsize=6)
                ax.axis("off")
            row[0].text(-0.05, 0.5, f"corr {r['corr']:.3f}\ndh256 {r['dhash256']}", transform=row[0].transAxes,
                        ha="right", va="center", fontsize=7)
        fig.suptitle(f"{tipo}: os {len(sel)} pares mais parecidos (esq. = teste)", fontsize=8)
        fig.tight_layout()
        fig.savefig(in_dir / name, dpi=100)
        plt.close(fig)
    print(f"\nSalvo em: {in_dir} (pares_refinados.csv, refinado_teste_x_treino.png, refinado_dentro_teste.png)")


if __name__ == "__main__":
    main()
