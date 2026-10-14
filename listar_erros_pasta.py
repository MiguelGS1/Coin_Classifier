import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from config import ARTIFACTS_DIR, DATASET_DIR
from dataset_loader import build_single_image_tensor

# Mostra, foto por foto, o que o modelo respondeu numa pasta organizada por classe (igual à do avaliar_pasta.py).
# Salva: uma planilha com todas as fotos, uma cópia das fotos ERRADAS separadas por tipo de erro e uma figura com
# as fotos erradas. Não altera a pasta original (só copia) e não treina nada.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lista as fotos que o modelo errou numa pasta por classe.")
    parser.add_argument("--modelo", required=True, help="Arquivo .keras")
    parser.add_argument("--pasta", required=True, help="Pasta com uma subpasta por classe")
    parser.add_argument("--saida", required=True, help="Pasta onde salvar a planilha, as cópias e a figura")
    return parser.parse_args()


def short(name: str) -> str:
    return name.split()[0]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.saida)
    if out_dir.exists():
        raise SystemExit(f"A pasta {out_dir} já existe. Use outra --saida.")
    class_names = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())
    folder = Path(args.pasta)
    model = tf.keras.models.load_model(args.modelo)

    rows = []
    for class_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        if class_dir.name not in class_names:
            raise SystemExit(f"Pasta '{class_dir.name}' não é uma das classes do modelo: {class_names}")
        true = class_names.index(class_dir.name)
        paths = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        for start in range(0, len(paths), 32):
            batch = paths[start:start + 32]
            probs = model(tf.concat([build_single_image_tensor(str(p)) for p in batch], axis=0), training=False).numpy()
            for path, p in zip(batch, probs):
                pred = int(np.argmax(p))
                rows.append({"arquivo": path, "verdadeira": true, "prevista": pred, "probs": p})
    if not rows:
        raise SystemExit(f"Nenhuma foto encontrada nas subpastas de {folder}")

    out_dir.mkdir(parents=True)
    with open(out_dir / "todas_as_fotos.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["arquivo", "moeda_certa", "resposta_do_modelo", "acertou"]
                        + [f"prob_{short(c)}" for c in class_names])
        for r in rows:
            writer.writerow([r["arquivo"].name, short(class_names[r["verdadeira"]]), short(class_names[r["prevista"]]),
                             "sim" if r["verdadeira"] == r["prevista"] else "NAO"]
                            + [f"{x * 100:.1f}" for x in r["probs"]])

    wrong = [r for r in rows if r["verdadeira"] != r["prevista"]]
    for r in wrong:  # cópia das fotos erradas, numa subpasta por tipo de erro
        target = out_dir / "erros" / f"{short(class_names[r['verdadeira']])}_virou_{short(class_names[r['prevista']])}"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r["arquivo"], target / r["arquivo"].name)

    if wrong:
        cols = 5
        lines = (len(wrong) + cols - 1) // cols
        fig, axes = plt.subplots(lines, cols, figsize=(2.6 * cols, 2.9 * lines), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, r in zip(axes.flat, wrong):
            ax.imshow(Image.open(r["arquivo"]).convert("RGB"))
            ax.set_title(f"{r['arquivo'].name[:22]}\ncerta: {short(class_names[r['verdadeira']])} | "
                         f"modelo: {short(class_names[r['prevista']])} ({r['probs'][r['prevista']] * 100:.0f}%)",
                         fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "fotos_erradas.png", dpi=120)
        plt.close(fig)

    print(f"\nFotos: {len(rows)}   acertos: {len(rows) - len(wrong)}   erros: {len(wrong)}")
    for r in wrong:
        print(f"  ERROU  {r['arquivo'].name}:  certa = {short(class_names[r['verdadeira']])}, "
              f"modelo disse {short(class_names[r['prevista']])} ({r['probs'][r['prevista']] * 100:.1f}%)")
    print(f"\nSalvo em: {out_dir}")
    print("  todas_as_fotos.csv (todas as fotos e as probabilidades), erros\\ (cópias das erradas),"
          " fotos_erradas.png (figura)")


if __name__ == "__main__":
    main()
