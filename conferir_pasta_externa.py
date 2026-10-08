import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from config import BASE_DIR

# Confere se as fotos de uma pasta externa (ex.: dataset/extra_ucoin) são cópias de fotos de outras pastas
# (ex.: dataset/raw, o U.S. Coins). Procura arquivos idênticos (MD5) e fotos quase iguais (hash visual dHash).
# Só lê as imagens; não altera nada.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
HASH_LIMIT = 6  # distância máxima do hash visual (0 a 64) para considerar duas fotos quase iguais


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Procura cópias de uma pasta externa em outras pastas.")
    parser.add_argument("--pasta", default=str(BASE_DIR / "dataset" / "extra_ucoin"))
    parser.add_argument("--comparar", nargs="+",
                        default=[str(BASE_DIR / "dataset" / "raw"), str(BASE_DIR / "dataset" / "controle_eua")])
    return parser.parse_args()


def images(folder: Path) -> list:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def fingerprints(path: Path) -> tuple:
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    small = np.asarray(Image.open(path).convert("L").resize((9, 8)), dtype=np.int16)
    return md5, (small[:, 1:] > small[:, :-1]).flatten()


def main() -> None:
    args = parse_args()
    target = images(Path(args.pasta))
    print(f"Fotos em {args.pasta}: {len(target)}")
    target_fp = [(p, *fingerprints(p)) for p in target]

    for other in args.comparar:
        others = images(Path(other))
        print(f"\nComparando com {other} ({len(others)} fotos)...")
        md5s, hashes = {}, []
        for p in others:
            md5, h = fingerprints(p)
            md5s.setdefault(md5, p)
            hashes.append((h, p))
        all_h = np.stack([h for h, _ in hashes]) if hashes else np.zeros((0, 64), bool)
        identical, similar = [], []
        for p, md5, h in target_fp:
            if md5 in md5s:
                identical.append((p, md5s[md5]))
                continue
            if len(all_h):
                dist = (all_h != h).sum(axis=1)
                best = int(dist.argmin())
                if dist[best] <= HASH_LIMIT:
                    similar.append((p, hashes[best][1], int(dist[best])))
        print(f"  Arquivos IDÊNTICOS: {len(identical)}")
        for a, b in identical[:10]:
            print(f"    {a.parent.name[:10]}/{a.name}  =  {b.parent.name[:10]}/{b.name}")
        print(f"  Fotos QUASE IGUAIS (hash visual <= {HASH_LIMIT}): {len(similar)}")
        for a, b, d in similar[:10]:
            print(f"    {a.parent.name[:10]}/{a.name}  ~  {b.parent.name[:10]}/{b.name}  (distância {d})")
    print("\nSe aparecer algo, abra os pares listados e confira se são mesmo a mesma foto.")


if __name__ == "__main__":
    main()
