import argparse
import json
from pathlib import Path

import numpy as np

from metrics import plot_confusion_matrix

# Refaz a figura da matriz de confusão do ensemble a partir do resultados_ensemble.json já salvo, com os nomes
# das moedas sem o período em inglês ("1938-Date"). Só lê o JSON; não roda nenhum modelo.
NOMES = ["Jefferson Nickels", "Lincoln Cents", "Washington Quarters"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refaz a matriz de confusão do ensemble com rótulos curtos.")
    parser.add_argument("--resultados", required=True, help="resultados_ensemble.json do avaliar_ensemble.py")
    parser.add_argument("--saida", required=True, help="Arquivo PNG novo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.saida)
    if out.exists():
        raise SystemExit(f"O arquivo {out} já existe. Use outro --saida.")
    data = json.loads(Path(args.resultados).read_text(encoding="utf-8"))
    classes = data["ensemble"]["classes"]
    if [c.split(",")[0] for c in classes] != NOMES:
        raise SystemExit(f"Ordem de classes inesperada: {classes}")
    matrix = np.array(data["ensemble"]["matriz_confusao"])
    plot_confusion_matrix(matrix, NOMES, out)
    print(f"Matriz (linhas = moeda certa):\n{matrix}\nFigura salva em: {out}")


if __name__ == "__main__":
    main()
