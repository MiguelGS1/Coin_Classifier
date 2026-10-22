import hashlib
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from config import BASE_DIR, DATASET_DIR, SEED
from io_utils import save_json
from make_group_split import series_key

# Divisão ALEATÓRIA por imagem, para comparar com a divisão por série (split_series_v2). Cada classe reserva para o
# teste o MESMO número de imagens do teste por série (151, 180 e 184), sorteadas com a semente fixa, sem olhar a
# série. O resto vai para treino_validacao, que os scripts de treino dividem em 80/20 como sempre.
# Só copia imagens: dataset/raw e dataset/split_series_v2 não são alterados.
OUTPUT_DIR = BASE_DIR / "dataset" / "split_aleatoria"
SERIES_TEST_DIR = BASE_DIR / "dataset" / "split_series_v2" / "teste"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT_DIR.exists():
        raise SystemExit(f"A pasta {OUTPUT_DIR} já existe. Para gerar a divisão de novo, apague essa pasta antes.")
    if not SERIES_TEST_DIR.exists():
        raise SystemExit(f"Não encontrei {SERIES_TEST_DIR}; ela é usada para ter o mesmo tamanho de teste.")

    rng = np.random.default_rng(SEED)
    class_dirs = sorted(p for p in DATASET_DIR.iterdir() if p.is_dir())
    report, lines = {}, []

    for class_dir in class_dirs:
        files = sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        n_test = sum(1 for p in (SERIES_TEST_DIR / class_dir.name).iterdir() if p.is_file())
        order = rng.permutation(len(files))
        test = [files[i] for i in order[:n_test]]
        train = [files[i] for i in order[n_test:]]

        for part, paths in (("treino_validacao", train), ("teste", test)):
            target_dir = OUTPUT_DIR / part / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for path in paths:
                shutil.copy2(path, target_dir / path.name)

        # Quanto o teste "conhece" do treino: imagens de séries que também estão no treino e cópias exatas
        train_series = Counter(series_key(p) for p in train)
        test_in_seen_series = sum(1 for p in test if train_series[series_key(p)] > 0)
        train_md5 = {md5(p) for p in train}
        test_exact_copies = sum(1 for p in test if md5(p) in train_md5)

        report[class_dir.name] = {
            "imagens_treino_validacao": len(train),
            "imagens_teste": len(test),
            "teste_de_series_tambem_no_treino": test_in_seen_series,
            "teste_com_copia_exata_no_treino": test_exact_copies,
        }
        lines.append(
            f"{class_dir.name:<32} treino+validação: {len(train):>5}   teste: {len(test):>4}   "
            f"teste de séries vistas: {test_in_seen_series:>4}   cópias exatas no treino: {test_exact_copies:>3}"
        )

    save_json({"semente": SEED, "classes": report}, OUTPUT_DIR / "divisao.json")
    print("Divisão aleatória concluída (as imagens foram copiadas; dataset/raw não foi alterado)\n")
    print("\n".join(lines))
    print(f"\nPasta criada: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
