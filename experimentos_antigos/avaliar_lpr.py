import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import ARTIFACTS_DIR, BASE_DIR, DATASET_DIR
from dataset_loader import build_single_image_tensor
from io_utils import save_json

# Usa os recortes do LPR (dataset/lpr_recortes) como TESTE À PARTE de um modelo já treinado. Não treina nada.
# O LPR mistura origens, então o resultado sai separado por grupo:
#  - "outras origens" e "cópias do uCoin": fotos que não estão no U.S. Coins -> teste externo de verdade;
#  - "cópias do U.S. Coins (séries do treino)": o modelo já viu essas fotos no treino -> NÃO vale como teste;
#  - "cópias do U.S. Coins (séries do teste)": as mesmas séries do teste v2 (pasta series_do_teste).
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
US_COINS = re.compile(r"^lpr_(Lincoln-Cent|Jefferson-Nickel|Washington-Quarter)-\d+_", re.IGNORECASE)
GROUPS = ["outras origens", "cópias do uCoin", "cópias do U.S. Coins (séries do treino)",
          "cópias do U.S. Coins (séries do teste)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia um modelo nos recortes do LPR, separado por origem.")
    parser.add_argument("--modelo", required=True, help="Arquivo .keras")
    parser.add_argument("--nome", required=True, help="Nome do resultado (ex.: lpr_balanco_cutout_s42)")
    parser.add_argument("--pasta", default=str(BASE_DIR / "dataset" / "lpr_recortes"))
    return parser.parse_args()


def group_of(path: Path, in_test_folder: bool) -> str:
    if in_test_folder:
        return GROUPS[3]
    if US_COINS.match(path.name):
        return GROUPS[2]
    if "usa" in path.name.lower():
        return GROUPS[1]
    return GROUPS[0]


def main() -> None:
    args = parse_args()
    out_path = ARTIFACTS_DIR / "experiments" / "avaliacoes" / f"{args.nome}.json"
    if out_path.exists():
        raise SystemExit(f"Já existe um resultado em {out_path}. Use outro --nome.")
    class_names = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())
    root = Path(args.pasta)
    items = []  # (caminho, classe verdadeira, grupo)
    for base, in_test in ((root, False), (root / "series_do_teste", True)):
        for i, cls in enumerate(class_names):
            folder = base / cls
            if folder.exists():
                items += [(p, i, group_of(p, in_test)) for p in sorted(folder.iterdir())
                          if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not items:
        raise SystemExit(f"Nenhuma imagem encontrada em {root}")

    model = tf.keras.models.load_model(args.modelo)
    preds = []
    for start in range(0, len(items), 32):
        batch = tf.concat([build_single_image_tensor(str(p)) for p, _, _ in items[start:start + 32]], axis=0)
        preds += list(np.argmax(model(batch, training=False).numpy(), axis=1))

    n = len(class_names)
    matrices = defaultdict(lambda: np.zeros((n, n), dtype=int))
    for (_, true, group), pred in zip(items, preds):
        matrices[group][true, pred] += 1
    external = matrices[GROUPS[0]] + matrices[GROUPS[1]]

    result = {"modelo": args.modelo, "pasta": str(root), "classes": class_names, "grupos": {}}
    print(f"\nModelo: {args.modelo}")
    print(f"{'grupo':<44}{'fotos':>7}{'acurácia':>10}{'balanc.':>9}" + "".join(f"{c.split()[0]:>12}" for c in class_names))
    rows = [(g, matrices[g]) for g in GROUPS] + [("TESTE EXTERNO (outras origens + uCoin)", external)]
    for name, m in rows:
        total = int(m.sum())
        if total == 0:
            continue
        present = m.sum(axis=1) > 0
        recall = [m[i, i] / m[i].sum() if present[i] else float("nan") for i in range(n)]
        acc = float(np.trace(m) / total)
        bal = float(np.nanmean(recall))
        result["grupos"][name] = {"fotos": total, "acuracia": acc, "balanceada": bal,
                                  "revocacao": {c: (None if np.isnan(r) else float(r)) for c, r in zip(class_names, recall)},
                                  "matriz_confusao": m.tolist()}
        print(f"{name:<44}{total:>7}{acc * 100:>9.1f}%{bal * 100:>8.1f}%"
              + "".join(f"{'-':>12}" if np.isnan(r) else f"{r * 100:>11.1f}%" for r in recall))
    print("\nMatriz do TESTE EXTERNO (linhas = verdadeira; colunas = " + ", ".join(c.split()[0] for c in class_names) + "):")
    for i, c in enumerate(class_names):
        print(f"  {c.split()[0]:<11} {external[i].tolist()}")
    print("\nAtenção: 'cópias do U.S. Coins (séries do treino)' são fotos que o modelo já viu no treino; não use como teste.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(result, out_path)
    print(f"Resultado salvo em: {out_path}")


if __name__ == "__main__":
    main()
