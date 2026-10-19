import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from config import ARTIFACTS_DIR, BASE_DIR
from make_group_split import series_key

# Monta "folhas de contato": todas as fotos de uma pasta, em miniatura, agrupadas por série, com o nome de cada foto
# embaixo. Serve para conferir a olho quais DESENHOS existem em cada série (ex.: Jefferson de 2005 em diante,
# Washington do bicentenário). Só lê as fotos; não altera nada.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Folhas de miniaturas por série, para inspeção visual.")
    parser.add_argument("--pasta", default=str(BASE_DIR / "dataset" / "split_series_v2" / "treino_validacao"))
    parser.add_argument("--classes", nargs="+", default=["Jefferson", "Washington"],
                        help="Início do nome das pastas de classe a incluir")
    parser.add_argument("--por_folha", type=int, default=80, help="Fotos por folha")
    parser.add_argument("--tamanho", type=int, default=150, help="Tamanho de cada miniatura (pixels)")
    parser.add_argument("--saida", default=str(ARTIFACTS_DIR / "experiments" / "folhas_por_serie"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root, out_dir = Path(args.pasta), Path(args.saida)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols, size, label_h = 10, args.tamanho, 14
    total = 0
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not any(class_dir.name.startswith(c) for c in args.classes):
            continue
        groups = defaultdict(list)
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                groups[series_key(p)].append(p)
        for serie, files in sorted(groups.items()):
            safe = serie.replace(" ", "_").replace("(", "").replace(")", "")
            for n, start in enumerate(range(0, len(files), args.por_folha), 1):
                chunk = files[start:start + args.por_folha]
                rows = (len(chunk) + cols - 1) // cols
                sheet = Image.new("RGB", (cols * size, rows * (size + label_h) + 22), "white")
                draw = ImageDraw.Draw(sheet)
                draw.text((4, 4), f"{serie} - folha {n} ({len(files)} fotos na serie)", fill="black")
                for i, path in enumerate(chunk):
                    thumb = Image.open(path).convert("RGB")
                    thumb.thumbnail((size, size))
                    x, y = (i % cols) * size, 22 + (i // cols) * (size + label_h)
                    sheet.paste(thumb, (x + (size - thumb.width) // 2, y + (size - thumb.height) // 2))
                    name = path.stem.replace(serie, "").strip(" _")
                    draw.text((x + 3, y + size), name[:22], fill="black")
                sheet.save(out_dir / f"{safe}_folha{n:02d}.jpg", quality=85)
                total += 1
            print(f"{serie}: {len(files)} fotos", flush=True)
    print(f"\n{total} folhas salvas em: {out_dir}")


if __name__ == "__main__":
    main()
