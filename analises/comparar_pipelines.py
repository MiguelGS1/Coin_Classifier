import argparse
import csv
from collections import Counter
from math import comb
from pathlib import Path

# Compara, FOTO POR FOTO, as duas configurações nas mesmas imagens:
#   A = ensemble sem recorte  (fotos_ensemble.csv, saída do avaliar_ensemble.py)
#   B = ensemble com recorte  (fotos_ensemble_recorte.csv, saída do avaliar_ensemble_recorte.py)
# Mostra quantas fotos só A acertou, só B acertou, as duas acertaram e as duas erraram; quais confusões
# (moeda certa -> resposta) sumiram ou apareceram com o recorte; e o teste exato de McNemar (as fotos em que só uma
# das duas acerta). Só lê os dois CSV; não carrega modelo nem imagem.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara foto por foto o ensemble sem e com recorte.")
    parser.add_argument("--csv_a", required=True, help="fotos_ensemble.csv (sem recorte)")
    parser.add_argument("--csv_b", required=True, help="fotos_ensemble_recorte.csv (com recorte)")
    parser.add_argument("--saida", required=True, help="Arquivo .csv onde salvar a comparação foto por foto")
    return parser.parse_args()


def read(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    return {(r["moeda_certa"], r["arquivo"]): r for r in rows}


def mcnemar_exact(b: int, c: int) -> float:
    # p bilateral do teste binomial exato nas fotos discordantes (só A acerta = b, só B acerta = c)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> None:
    args = parse_args()
    a, b = read(args.csv_a), read(args.csv_b)
    if set(a) != set(b):
        so_a, so_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        raise SystemExit(f"Os dois CSV não têm as mesmas fotos (só em A: {len(so_a)}, só em B: {len(so_b)}). "
                         f"Exemplos: {so_a[:3]} {so_b[:3]}")
    keys = sorted(a)
    n = len(keys)

    groups = {"as duas acertam": [], "só A acerta": [], "só B acerta": [], "as duas erram": []}
    for k in keys:
        ok_a = a[k]["ensemble"] == k[0]
        ok_b = b[k]["ensemble"] == k[0]
        name = ("as duas acertam" if ok_a and ok_b else "só A acerta" if ok_a else
                "só B acerta" if ok_b else "as duas erram")
        groups[name].append(k)

    acc_a = sum(a[k]["ensemble"] == k[0] for k in keys) / n
    acc_b = sum(b[k]["ensemble"] == k[0] for k in keys) / n
    print(f"Fotos: {n}")
    print(f"  A (sem recorte): {acc_a * 100:.1f}%   B (com recorte): {acc_b * 100:.1f}%")
    print("\nComparação pareada (mesmas fotos):")
    for name, items in groups.items():
        print(f"  {name:<18}{len(items):>5}  ({len(items) / n * 100:.1f}%)")
    p = mcnemar_exact(len(groups["só A acerta"]), len(groups["só B acerta"]))
    print(f"  Teste exato de McNemar (só A × só B): p = {p:.3f}"
          f"  ({'diferença significativa' if p < 0.05 else 'diferença NÃO significativa'} a 5%)")

    print("\nConfusões do ensemble (moeda certa -> resposta):")
    conf_a = Counter((k[0], a[k]["ensemble"]) for k in keys if a[k]["ensemble"] != k[0])
    conf_b = Counter((k[0], b[k]["ensemble"]) for k in keys if b[k]["ensemble"] != k[0])
    print(f"  {'confusão':<26}{'A':>5}{'B':>5}")
    for c in sorted(set(conf_a) | set(conf_b)):
        print(f"  {c[0] + ' -> ' + c[1]:<26}{conf_a[c]:>5}{conf_b[c]:>5}")
    print(f"  {'total de erros':<26}{sum(conf_a.values()):>5}{sum(conf_b.values()):>5}")

    print("\nPor moeda (acurácia do ensemble):")
    for coin in sorted({k[0] for k in keys}):
        ks = [k for k in keys if k[0] == coin]
        ra = sum(a[k]["ensemble"] == coin for k in ks) / len(ks)
        rb = sum(b[k]["ensemble"] == coin for k in ks) / len(ks)
        print(f"  {coin:<12} A {ra * 100:5.1f}%   B {rb * 100:5.1f}%   ({len(ks)} fotos)")

    for name in ("só A acerta", "só B acerta", "as duas erram"):
        if groups[name]:
            print(f"\n{name.upper()}:")
            for k in groups[name]:
                print(f"  {k[1]:<45} certa: {k[0]:<11} A: {a[k]['ensemble']:<11} B: {b[k]['ensemble']}")

    out = Path(args.saida)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["arquivo", "moeda_certa", "grupo", "resposta_A", "resposta_B"])
        for name, items in groups.items():
            for k in items:
                writer.writerow([k[1], k[0], name, a[k]["ensemble"], b[k]["ensemble"]])
    print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    main()
