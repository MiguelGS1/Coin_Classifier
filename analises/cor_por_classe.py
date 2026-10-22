import argparse
import json

# Acerto do ensemble por moeda, na imagem original e em tons de cinza, sem e com recorte, a partir dos resultados
# já salvos pelo oclusao_ensemble.py (resultados_oclusao_ensemble.json). Mostra também quantas imagens cada moeda
# ganhou ou perdeu sem a cor, para separar os efeitos que se compensam na acurácia total. Só lê os arquivos;
# não roda nenhum modelo.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acerto por moeda em tons de cinza, sem e com recorte.")
    parser.add_argument("--sem_recorte", required=True, help="resultados_oclusao_ensemble.json do teste sem recorte")
    parser.add_argument("--com_recorte", required=True, help="resultados_oclusao_ensemble.json do teste com recorte")
    return parser.parse_args()


def per_class(path: str) -> tuple[list, dict]:
    data = json.loads(open(path, encoding="utf-8").read())
    classes, out = data["classes"], {}
    for test in ("original", "tons_de_cinza"):
        matrix = data["resultados"][test]["ensemble"]["matriz_confusao"]
        out[test] = {c: (matrix[i][i], sum(matrix[i])) for i, c in enumerate(classes)}
        out[test]["Total"] = (sum(matrix[i][i] for i in range(len(classes))), sum(map(sum, matrix)))
    return classes + ["Total"], out


def pct(v: tuple) -> str:
    return f"{100 * v[0] / v[1]:.1f}".replace(".", ",")


def main() -> None:
    args = parse_args()
    rows, a = per_class(args.sem_recorte)
    _, b = per_class(args.com_recorte)
    print(f"{'Moeda':<34}{'Imagens':>8}  {'Sem recorte: orig. -> cinza':>32}  {'Com recorte: orig. -> cinza':>32}")
    latex = []
    for c in rows:
        name = c.split(",")[0]
        n = a["original"][c][1]
        cells = []
        for r in (a, b):
            o, g = r["original"][c], r["tons_de_cinza"][c]
            cells.append(f"{pct(o):>6}% -> {pct(g):>6}% ({g[0] - o[0]:+d} imagens)")
        print(f"{name:<34}{n:>8}  {cells[0]:>32}  {cells[1]:>32}")
        latex.append(f"{name} & {n} & {pct(a['original'][c])}\\% & {pct(a['tons_de_cinza'][c])}\\% & "
                     f"{pct(b['original'][c])}\\% & {pct(b['tons_de_cinza'][c])}\\% \\\\")
    print("\nLinhas para a tabela do artigo (LaTeX):")
    print("\n".join(latex))


if __name__ == "__main__":
    main()
