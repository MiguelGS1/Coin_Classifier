import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import ARTIFACTS_DIR, SEED
from io_utils import save_json
from make_group_split import series_key

# Acerto do ensemble por SÉRIE do teste, sem e com recorte, e intervalos de confiança de 95%:
# - por imagem (Wilson), que trata cada foto como independente;
# - por série (bootstrap estratificado por moeda: sorteia, com reposição, as séries de cada moeda e recalcula a
#   acurácia total), que respeita o fato de as fotos de uma mesma série serem parecidas entre si.
# No World Coins, que não tem séries, só o intervalo por imagem. Só lê os CSV das avaliações; não roda nenhum modelo.
N_BOOT = 10000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acerto por série e intervalos de confiança.")
    parser.add_argument("--sem_recorte", required=True, help="fotos_ensemble.csv do teste")
    parser.add_argument("--com_recorte", required=True, help="fotos_ensemble_recorte.csv do teste")
    parser.add_argument("--world_sem", default=None, help="fotos_ensemble.csv do World Coins (opcional)")
    parser.add_argument("--world_com", default=None, help="fotos_ensemble_recorte.csv do World Coins (opcional)")
    parser.add_argument("--saida", default=str(ARTIFACTS_DIR / "experiments" / "resultados_por_serie.json"))
    return parser.parse_args()


def read(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as fh:
        return {r["arquivo"]: r for r in csv.DictReader(fh, delimiter=";")}


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def pct(x: float) -> str:
    return f"{x * 100:.1f}".replace(".", ",")


def main() -> None:
    args = parse_args()
    a, b = read(args.sem_recorte), read(args.com_recorte)
    if set(a) != set(b):
        raise SystemExit("Os dois CSV não têm as mesmas fotos.")

    # Acertos por série: [fotos, acertos sem recorte, acertos com recorte], agrupados por moeda
    series = defaultdict(lambda: [0, 0, 0])
    coin_of = {}
    for name, row in a.items():
        key = series_key(Path(name))
        coin_of[key] = row["moeda_certa"]
        s = series[key]
        s[0] += 1
        s[1] += row["acertou"] == "sim"
        s[2] += b[name]["acertou"] == "sim"

    print(f"{'Série':<36}{'Moeda':<12}{'Fotos':>6}{'Sem recorte':>14}{'Com recorte':>14}")
    table, latex = [], []
    for key in sorted(series, key=lambda k: (coin_of[k], k)):
        n, ka, kb = series[key]
        table.append({"serie": key, "moeda": coin_of[key], "fotos": n, "acertos_sem": ka, "acertos_com": kb})
        print(f"{key:<36}{coin_of[key]:<12}{n:>6}{pct(ka / n):>13}%{pct(kb / n):>13}%")
        latex.append(f"{key} & {n} & {pct(ka / n)}\\% & {pct(kb / n)}\\% \\\\")

    # Bootstrap estratificado por moeda, sorteando séries inteiras
    rng = np.random.default_rng(SEED)
    by_coin = defaultdict(list)
    for key, v in series.items():
        by_coin[coin_of[key]].append(v)
    boot = {"sem": [], "com": [], "diferenca": []}
    for _ in range(N_BOOT):
        tot = np.zeros(3)
        for groups in by_coin.values():
            idx = rng.integers(0, len(groups), len(groups))
            tot += np.array([groups[i] for i in idx]).sum(axis=0)
        boot["sem"].append(tot[1] / tot[0])
        boot["com"].append(tot[2] / tot[0])
        boot["diferenca"].append((tot[2] - tot[1]) / tot[0])
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in boot.items()}

    n_total = len(a)
    k_sem = sum(r["acertou"] == "sim" for r in a.values())
    k_com = sum(r["acertou"] == "sim" for r in b.values())
    summary = {
        "series": table,
        "teste": {
            "sem_recorte": {"acuracia": k_sem / n_total, "ic_por_imagem_wilson": wilson(k_sem, n_total),
                            "ic_por_serie_bootstrap": ci["sem"]},
            "com_recorte": {"acuracia": k_com / n_total, "ic_por_imagem_wilson": wilson(k_com, n_total),
                            "ic_por_serie_bootstrap": ci["com"]},
            "diferenca_com_menos_sem": {"valor": (k_com - k_sem) / n_total,
                                        "ic_por_serie_bootstrap": ci["diferenca"]},
            "reamostragens": N_BOOT, "semente": SEED,
        },
    }
    print("\nIntervalos de confiança de 95% no teste:")
    for cfg, k in (("sem_recorte", k_sem), ("com_recorte", k_com)):
        s = summary["teste"][cfg]
        w, bt = s["ic_por_imagem_wilson"], s["ic_por_serie_bootstrap"]
        print(f"  {cfg:<12} {pct(k / n_total)}%   por imagem: {pct(w[0])}–{pct(w[1])}%   "
              f"por série: {pct(bt[0])}–{pct(bt[1])}%")
    d = summary["teste"]["diferenca_com_menos_sem"]
    pts = lambda x: f"{x * 100:+.1f}".replace(".", ",")
    print(f"  diferença (com − sem): {pts(d['valor'])} pontos   por série: "
          f"{pts(ci['diferenca'][0])} a {pts(ci['diferenca'][1])} pontos")

    if args.world_sem and args.world_com:
        summary["world_coins"] = {}
        print("\nWorld Coins (intervalo por imagem, Wilson):")
        for cfg, path in (("sem_recorte", args.world_sem), ("com_recorte", args.world_com)):
            rows = read(path)
            k, n = sum(r["acertou"] == "sim" for r in rows.values()), len(rows)
            lo, hi = wilson(k, n)
            summary["world_coins"][cfg] = {"acertos": k, "fotos": n, "acuracia": k / n, "ic_por_imagem_wilson": (lo, hi)}
            print(f"  {cfg:<12} {k}/{n} = {pct(k / n)}%   IC 95%: {pct(lo)}–{pct(hi)}%")

    print("\nLinhas para a tabela por série (LaTeX):")
    print("\n".join(latex))
    save_json(summary, Path(args.saida))
    print(f"\nResultados salvos em: {args.saida}")


if __name__ == "__main__":
    main()
