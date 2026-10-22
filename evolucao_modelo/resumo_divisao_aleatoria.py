import json

import numpy as np

from config import ARTIFACTS_DIR
from io_utils import save_json

# Junta os 6 treinos com a divisão ALEATÓRIA (modelo inicial e final, sementes 42, 1 e 2) numa tabela, com a média
# e o desvio padrão das três sementes (o mesmo desvio, com n - 1, usado no restante do trabalho). Só lê os
# resultados.json; não treina nada.
EXPERIMENTS = ARTIFACTS_DIR / "experiments"
SEEDS = (42, 1, 2)
MODELS = {"inicial": "Inicial (4 blocos, parada antecipada)", "final": "Final (6 blocos + balanço de branco + cutout, 40 épocas)"}


def main() -> None:
    summary, missing = {}, []
    for key, label in MODELS.items():
        rows = {}
        for seed in SEEDS:
            path = EXPERIMENTS / f"aleatoria_{key}_s{seed}" / "resultados.json"
            if not path.exists():
                missing.append(str(path))
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            rows[seed] = {
                "teste": data["teste_series_novas"]["acuracia"] * 100,
                "f1_teste": data["teste_series_novas"]["f1_macro"],
                "validacao": data["validacao_series_vistas"]["acuracia"] * 100,
                "epocas": data["epocas_executadas"],
            }
        if not rows:
            continue
        test = np.array([r["teste"] for r in rows.values()])
        val = np.array([r["validacao"] for r in rows.values()])
        summary[key] = {
            "modelo": label,
            "sementes": rows,
            "teste_media": float(test.mean()),
            "teste_desvio": float(test.std(ddof=1)) if len(test) > 1 else None,
            "validacao_media": float(val.mean()),
            "validacao_desvio": float(val.std(ddof=1)) if len(val) > 1 else None,
        }

    if missing:
        print("Ainda faltam estes treinos:\n  " + "\n  ".join(missing) + "\n")
    print(f"{'Modelo':<60}{'Semente':>8}{'Teste':>9}{'Validação':>11}{'Épocas':>8}")
    for key, s in summary.items():
        for seed, r in s["sementes"].items():
            print(f"{s['modelo']:<60}{seed:>8}{r['teste']:>8.1f}%{r['validacao']:>10.1f}%{r['epocas']:>8}")
        sd = f" ± {s['teste_desvio']:.1f}" if s["teste_desvio"] is not None else ""
        print(f"{'  média':<60}{'':>8}{s['teste_media']:>8.1f}%{sd}\n")
    out = EXPERIMENTS / "resumo_divisao_aleatoria.json"
    save_json(summary, out)
    print(f"Resumo salvo em: {out}")


if __name__ == "__main__":
    main()
