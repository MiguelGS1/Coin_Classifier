import hashlib
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import BASE_DIR, DATASET_DIR, SEED
from io_utils import save_json

OUTPUT_DIR = BASE_DIR / "dataset" / "split_series_v2"
TEST_FRACTION = 0.15
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
FACE_WORDS = (" obverse", " reverse")
# Nomes com dois números ("Washington Quarter 11 6_2.jpg") são uma única coleção de fotos (mesmo estilo)
TWO_NUMBERS = re.compile(r"^(?P<moeda>.*?) \d+ \d+$")


def series_key(path: Path) -> str:
    # "Lincoln Cent 37633_208 Obverse.jpg" -> "Lincoln Cent 37633"
    # O número antes do último "_" identifica a série (origem) das fotografias
    stem = path.stem
    for word in FACE_WORDS:
        if stem.lower().endswith(word):
            stem = stem[: -len(word)]
    if "_" in stem:
        stem = stem.rsplit("_", 1)[0]
    match = TWO_NUMBERS.match(stem.strip())
    if match:
        return f"{match['moeda']} (coleção dois números)"
    return stem.strip()


def build_groups(files: list) -> dict:
    # Grupo = série; séries que têm algum arquivo idêntico entre si são unidas num grupo só,
    # para que cópias exatas nunca fiquem uma no treino e outra no teste
    parent = {}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    first_series_of_hash = {}
    for path in files:
        key = series_key(path)
        parent.setdefault(key, key)
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in first_series_of_hash:
            a, b = find(first_series_of_hash[digest]), find(key)
            if a != b:
                parent[b] = a
        else:
            first_series_of_hash[digest] = key

    groups = defaultdict(list)
    for path in files:
        groups[find(series_key(path))].append(path)
    return groups


def split_class(files: list, rng: np.random.Generator) -> tuple[list, list]:
    groups = build_groups(files)
    # Sorteia a ordem dos grupos e reserva grupos inteiros para o teste até chegar a ~15% da classe
    keys = sorted(groups)
    rng.shuffle(keys)
    target = TEST_FRACTION * len(files)
    test_keys, test_count = [], 0
    for key in keys:
        if test_count >= target:
            break
        size = len(groups[key])
        if test_count + size <= target * 1.1:  # pula grupos grandes demais para o espaço que resta
            test_keys.append(key)
            test_count += size
    train_keys = [k for k in keys if k not in test_keys]
    return [(k, groups[k]) for k in train_keys], [(k, groups[k]) for k in test_keys]


def main() -> None:
    if OUTPUT_DIR.exists():
        raise SystemExit(
            f"A pasta {OUTPUT_DIR} já existe. Para gerar a divisão de novo, apague essa pasta antes."
        )

    rng = np.random.default_rng(SEED)
    class_dirs = sorted(p for p in DATASET_DIR.iterdir() if p.is_dir())
    report, lines = {}, []

    for class_dir in class_dirs:
        files = sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        train_groups, test_groups = split_class(files, rng)

        for part, groups in (("treino_validacao", train_groups), ("teste", test_groups)):
            target_dir = OUTPUT_DIR / part / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for _, paths in groups:
                for path in paths:
                    shutil.copy2(path, target_dir / path.name)

        n_train = sum(len(p) for _, p in train_groups)
        n_test = sum(len(p) for _, p in test_groups)
        report[class_dir.name] = {
            "grupos_treino_validacao": len(train_groups),
            "grupos_teste": len(test_groups),
            "imagens_treino_validacao": n_train,
            "imagens_teste": n_test,
            "grupos_do_teste": [k for k, _ in test_groups],
        }
        lines.append(
            f"{class_dir.name:<32} treino+validação: {n_train:>5} imagens ({len(train_groups):>3} grupos)   "
            f"teste: {n_test:>4} imagens ({len(test_groups):>3} grupos, {n_test / len(files) * 100:.1f}%)"
        )

    save_json(report, OUTPUT_DIR / "divisao.json")
    print("Divisão por série concluída (as imagens foram copiadas; dataset/raw não foi alterado)")
    print("Séries com arquivos idênticos entre si foram mantidas no mesmo lado.\n")
    print("\n".join(lines))
    print(f"\nPasta criada: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
