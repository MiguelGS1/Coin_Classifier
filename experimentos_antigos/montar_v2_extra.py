import argparse
import shutil
from pathlib import Path

from config import BASE_DIR

# Monta uma pasta nova em que treino_validacao = treino_validacao da v2 + imagens extras (uma subpasta por classe);
# teste = o MESMO teste da v2 (sem nenhuma foto nova). Só copia; não altera as pastas de origem.
V2_DIR = BASE_DIR / "dataset" / "split_series_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Junta imagens extras ao treino da divisão v2.")
    parser.add_argument("--extra", default=str(BASE_DIR / "dataset" / "wikimedia_original"),
                        help="Pasta com as imagens extras, uma subpasta por classe")
    parser.add_argument("--saida", default=str(BASE_DIR / "dataset" / "split_series_v2_wikimedia"))
    return parser.parse_args()


def copy_tree(source: Path, target: Path) -> int:
    count = 0
    for path in source.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            out = target / path.relative_to(source)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
            count += 1
    return count


def main() -> None:
    args = parse_args()
    EXTRA_DIR, OUTPUT_DIR = Path(args.extra), Path(args.saida)
    if OUTPUT_DIR.exists():
        raise SystemExit(f"A pasta {OUTPUT_DIR} já existe. Apague-a para montar de novo.")
    for folder in (V2_DIR, EXTRA_DIR):
        if not folder.exists():
            raise SystemExit(f"Pasta não encontrada: {folder}")

    print("Imagens por moeda (treino_validacao):")
    for class_dir in sorted(p for p in (V2_DIR / "treino_validacao").iterdir() if p.is_dir()):
        n_v2 = copy_tree(class_dir, OUTPUT_DIR / "treino_validacao" / class_dir.name)
        extra = EXTRA_DIR / class_dir.name  # a subpasta "parecidas_com_teste" NÃO entra
        n_extra = copy_tree(extra, OUTPUT_DIR / "treino_validacao" / class_dir.name) if extra.exists() else 0
        print(f"  {class_dir.name:<32} v2: {n_v2:>5}   extras: {n_extra:>5}   total: {n_v2 + n_extra:>5}")
    n_test = copy_tree(V2_DIR / "teste", OUTPUT_DIR / "teste")
    print(f"\nTeste: {n_test} imagens (o mesmo teste da v2)")
    print(f"Pasta criada: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
