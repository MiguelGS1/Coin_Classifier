import argparse
import csv
from collections import Counter
from pathlib import Path

# Lê o fotos_ensemble.csv criado pelo avaliar_ensemble.py e separa as fotos em grupos:
#   - todos acertam;  - discordam (os modelos não deram a mesma resposta);
#   - o ensemble CORRIGE (algum modelo errou, o ensemble acertou);  - o ensemble PIORA (algum acertou, o ensemble errou);
#   - todos erram (erro que o ensemble não tem como corrigir).
# Também conta, para cada modelo, quantos erros dele o ensemble corrigiu e quantos erros novos apareceram.
# Só lê o CSV; não carrega modelo nem imagem.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analisa concordância e erros do ensemble (fotos_ensemble.csv).")
    parser.add_argument("--csv", required=True, help="fotos_ensemble.csv (saída do avaliar_ensemble.py)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.csv)
    with open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    models = [c[: -len("_resposta")] for c in rows[0] if c.endswith("_resposta")]

    groups = {"todos acertam": [], "todos erram": [], "ensemble corrige": [], "ensemble piora": []}
    disagree = 0
    per_model = {m: Counter() for m in models}
    for r in rows:
        answers = [r[f"{m}_resposta"] for m in models]
        right = [a == r["moeda_certa"] for a in answers]
        ens_right = r["ensemble"] == r["moeda_certa"]
        disagree += len(set(answers)) > 1
        if all(right):
            groups["todos acertam"].append(r)
        elif not any(right):
            groups["todos erram"].append(r)
        elif ens_right:
            groups["ensemble corrige"].append(r)
        else:
            groups["ensemble piora"].append(r)
        for m, ok in zip(models, right):
            if not ok and ens_right:
                per_model[m]["corrigidos"] += 1
            if ok and not ens_right:
                per_model[m]["novos"] += 1
            per_model[m]["erros"] += not ok

    n = len(rows)
    print(f"Arquivo: {path}   fotos: {n}   modelos: {', '.join(models)}")
    print(f"\nOs modelos discordam em {disagree} fotos ({disagree / n * 100:.1f}%)")
    for name, items in groups.items():
        print(f"  {name:<18}{len(items):>5}  ({len(items) / n * 100:.1f}%)")
    print(f"  (erros do ensemble = 'todos erram' + 'ensemble piora' = "
          f"{len(groups['todos erram']) + len(groups['ensemble piora'])})")

    print(f"\n{'Modelo':<10}{'erros':>7}{'corrigidos pelo ensemble':>27}{'erros novos do ensemble':>26}")
    for m in models:
        c = per_model[m]
        print(f"{m:<10}{c['erros']:>7}{c['corrigidos']:>27}{c['novos']:>26}")

    for name in ("todos erram", "ensemble piora", "ensemble corrige"):
        if groups[name]:
            print(f"\n{name.upper()}:")
            for r in groups[name]:
                answers = " / ".join(r[f"{m}_resposta"] for m in models)
                print(f"  {r['arquivo']:<45} certa: {r['moeda_certa']:<11} modelos: {answers:<35} ensemble: {r['ensemble']}")

    out = path.with_name("analise_ensemble.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["arquivo", "moeda_certa", "grupo", "ensemble"] + [f"{m}_resposta" for m in models])
        for name, items in groups.items():
            for r in items:
                writer.writerow([r["arquivo"], r["moeda_certa"], name, r["ensemble"]]
                                + [r[f"{m}_resposta"] for m in models])
    print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    main()
