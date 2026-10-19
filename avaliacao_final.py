import argparse
import csv
import hashlib
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from config import DATASET_DIR, IMAGE_SIZE
from dataset_loader import build_single_image_tensor
from io_utils import save_json
from metrics import build_confusion_matrix, compute_metrics, plot_confusion_matrix
from prever_moeda_recortada import isolate_coin, load_rgb

# AVALIAÇÃO FINAL: roda UMA VEZ as duas configurações congeladas nas MESMAS fotos novas e salva tudo.
#   A = ensemble sem recorte  (mesmo pré-processamento do avaliar_ensemble.py: build_single_image_tensor)
#   B = ensemble com recorte  (mesmo pré-processamento do avaliar_ensemble_recorte.py: isolate_coin + resize)
# Antes de classificar, confere se as pastas e as planilhas batem (toda foto tem linha e toda linha tem foto). Se algo
# não bater, para SEM classificar nenhuma foto. Nenhuma foto é descartada por causa da previsão.
# Relata: acurácia e F1 macro (ensemble e cada modelo), matriz de confusão, resultados por moeda (exemplar), por
# sessão, por face e por acabamento, a comparação foto a foto A x B com o teste exato de McNemar, e o SHA-256 dos
# modelos e scripts usados. A pasta de saída não pode existir (para não sobrescrever uma execução anterior).
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCRIPTS = ["config.py", "dataset_loader.py", "metrics.py", "prever_moeda_recortada.py", "avaliacao_final.py"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avaliação final das duas configurações (sem e com recorte).")
    parser.add_argument("--modelos", nargs="+", required=True, help="Arquivos .keras (as 3 sementes)")
    parser.add_argument("--nomes", nargs="+", required=True, help="Um nome curto por modelo (ex.: s42 s1 s2)")
    parser.add_argument("--pasta", required=True, help="dataset/teste_final (uma subpasta por classe)")
    parser.add_argument("--fotos_csv", required=True, help="fotos_teste_final.csv")
    parser.add_argument("--exemplares_csv", required=True, help="exemplares_teste_final.csv")
    parser.add_argument("--saida", required=True, help="Pasta nova onde salvar os resultados")
    parser.add_argument("--conferir", action="store_true",
                        help="Só confere pastas e planilhas, sem carregar modelos nem classificar nada")
    return parser.parse_args()


def short(name: str) -> str:
    return name.split()[0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: str) -> list:
    with open(path, encoding="utf-8-sig") as fh:
        return [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(fh, delimiter=";")]


def mcnemar_exact(only_a: int, only_b: int) -> float:
    n = only_a + only_b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n)


def check_inputs(args, class_names):
    # Confere tudo ANTES de classificar. Devolve a lista de fotos na ordem das planilhas.
    problems = []
    folder = Path(args.pasta)
    photos, exemplars = read_csv(args.fotos_csv), read_csv(args.exemplares_csv)
    by_code = {e["exemplar"]: e for e in exemplars}
    if len(by_code) != len(exemplars):
        problems.append("Há código de exemplar repetido na planilha de exemplares.")
    files = {}
    for class_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        if class_dir.name not in class_names:
            problems.append(f"Pasta '{class_dir.name}' não é uma das classes: {class_names}")
            continue
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                files[p.name] = p
    names = [r["arquivo"] for r in photos]
    for n in sorted({n for n in names if names.count(n) > 1}):
        problems.append(f"Foto repetida na planilha: {n}")
    for n in sorted(set(files) - set(names)):
        problems.append(f"Foto na pasta sem linha na planilha: {n}")
    for n in sorted(set(names) - set(files)):
        problems.append(f"Linha na planilha sem foto na pasta: {n}")
    items = []
    for r in photos:
        path = files.get(r["arquivo"])
        if path is None:
            continue
        folder_class = path.parent.name
        if short(folder_class).lower() != r["classe"].lower():
            problems.append(f"{r['arquivo']}: classe '{r['classe']}' na planilha, mas está na pasta '{folder_class}'")
        ex = by_code.get(r["exemplar"])
        if ex is None:
            problems.append(f"{r['arquivo']}: exemplar '{r['exemplar']}' não está na planilha de exemplares")
        elif ex["classe"].lower() != r["classe"].lower():
            problems.append(f"{r['arquivo']}: classe diferente da do exemplar {r['exemplar']}")
        items.append({**r, "caminho": path, "rotulo": class_names.index(folder_class),
                      "acabamento": (ex or {}).get("acabamento", "")})
    return items, by_code, problems


def group_table(items, key, ok_a, ok_b):
    groups = defaultdict(list)
    for i, it in enumerate(items):
        groups[it[key] or "(vazio)"].append(i)
    return {g: {"fotos": len(ix), "acertos_A": int(ok_a[ix].sum()), "acertos_B": int(ok_b[ix].sum()),
                "acuracia_A": float(ok_a[ix].mean()), "acuracia_B": float(ok_b[ix].mean())}
            for g, ix in sorted(groups.items())}


def main() -> None:
    args = parse_args()
    if len(args.modelos) != len(args.nomes):
        raise SystemExit("Passe um nome para cada modelo (--nomes).")
    out_dir = Path(args.saida)
    if out_dir.exists() and not args.conferir:
        raise SystemExit(f"A pasta {out_dir} já existe. A avaliação final roda uma única vez; use outra --saida.")
    class_names = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())

    items, exemplars, problems = check_inputs(args, class_names)
    print(f"Fotos: {len(items)} | Exemplares: {len(exemplars)}", flush=True)
    if problems:
        print("\nPROBLEMAS (nada foi classificado):")
        for p in problems:
            print("  - " + p)
        raise SystemExit(1)
    print("Pastas e planilhas conferem.", flush=True)
    if args.conferir:
        print("Modo --conferir: nenhuma foto foi classificada.")
        return

    labels = np.array([it["rotulo"] for it in items])
    out_dir.mkdir(parents=True)
    crops_dir = out_dir / "recortes"

    print("Preparando as imagens das duas configurações...", flush=True)
    tensors_a, tensors_b, found = [], [], []
    for it in items:
        tensors_a.append(build_single_image_tensor(str(it["caminho"])))
        coin, ok = isolate_coin(load_rgb(str(it["caminho"])), 255)
        found.append(ok)
        target = crops_dir / it["caminho"].parent.name / f"{it['caminho'].stem}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(coin).save(target, quality=90)
        tensors_b.append(tf.image.resize(tf.constant(coin[None], dtype=tf.float32), IMAGE_SIZE))
    found = np.array(found)

    probs = {"A": [], "B": []}  # configuração -> [modelo, foto, classe]
    for model_path, nome in zip(args.modelos, args.nomes):
        print(f"Classificando com {nome} ...", flush=True)
        model = tf.keras.models.load_model(model_path)
        for cfg, tensors in (("A", tensors_a), ("B", tensors_b)):
            probs[cfg].append(np.concatenate(
                [model(tf.concat(tensors[s:s + 32], axis=0), training=False).numpy()
                 for s in range(0, len(tensors), 32)]))
    probs = {cfg: np.stack(p) for cfg, p in probs.items()}

    results = {"modelos": {n: {"arquivo": m, "sha256": sha256(Path(m))} for n, m in zip(args.nomes, args.modelos)},
               "scripts": {s: sha256(Path(s)) for s in SCRIPTS if Path(s).exists()},
               "pasta": args.pasta, "total_fotos": len(items), "total_exemplares": len(exemplars),
               "circulo_encontrado": int(found.sum()), "configuracoes": {}}
    preds = {}
    print(f"\n{'':<22}{'Sem recorte (A)':>18}{'Com recorte (B)':>18}")
    for cfg, nome_cfg in (("A", "sem_recorte"), ("B", "com_recorte")):
        indiv = {}
        for nome, p in zip(args.nomes, probs[cfg]):
            indiv[nome] = compute_metrics(build_confusion_matrix(labels, p.argmax(axis=1), len(class_names)),
                                          class_names)
        accs = [indiv[n]["acuracia"] for n in args.nomes]
        preds[cfg] = probs[cfg].mean(axis=0).argmax(axis=1)
        matrix = build_confusion_matrix(labels, preds[cfg], len(class_names))
        ens = compute_metrics(matrix, class_names)
        results["configuracoes"][nome_cfg] = {
            "individual": indiv, "media_individual": float(np.mean(accs)),
            "desvio_individual": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0, "ensemble": ens}
        plot_confusion_matrix(matrix, class_names, out_dir / f"matriz_confusao_{nome_cfg}.png")
    ra, rb = results["configuracoes"]["sem_recorte"], results["configuracoes"]["com_recorte"]
    for nome in args.nomes:
        print(f"{nome:<22}{ra['individual'][nome]['acuracia'] * 100:>17.1f}%"
              f"{rb['individual'][nome]['acuracia'] * 100:>17.1f}%")
    print(f"{'média ± DP':<22}{ra['media_individual'] * 100:>10.1f} ± {ra['desvio_individual'] * 100:.1f}%"
          f"{rb['media_individual'] * 100:>10.1f} ± {rb['desvio_individual'] * 100:.1f}%")
    print(f"{'ENSEMBLE':<22}{ra['ensemble']['acuracia'] * 100:>17.1f}%{rb['ensemble']['acuracia'] * 100:>17.1f}%")
    print(f"{'F1 macro do ensemble':<22}{ra['ensemble']['f1_macro']:>18.3f}{rb['ensemble']['f1_macro']:>18.3f}")
    print(f"Círculo encontrado em {found.sum()} de {len(items)} fotos "
          f"({len(items) - found.sum()} usaram o quadrado central)")

    ok_a, ok_b = preds["A"] == labels, preds["B"] == labels
    only_a, only_b = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    results["comparacao_pareada"] = {
        "as_duas_acertam": int((ok_a & ok_b).sum()), "so_A_acerta": only_a, "so_B_acerta": only_b,
        "as_duas_erram": int((~ok_a & ~ok_b).sum()), "p_mcnemar_exato": mcnemar_exact(only_a, only_b)}
    cp = results["comparacao_pareada"]
    print(f"\nComparação foto a foto: as duas acertam {cp['as_duas_acertam']} | só A {only_a} | só B {only_b} | "
          f"as duas erram {cp['as_duas_erram']} | McNemar exato p = {cp['p_mcnemar_exato']:.3f}")

    for key, title in (("exemplar", "moeda (exemplar)"), ("sessao", "sessão"), ("face", "face"),
                       ("acabamento", "acabamento")):
        table = group_table(items, key, ok_a, ok_b)
        results[f"por_{key}"] = table
        print(f"\nPor {title}:")
        for g, v in table.items():
            print(f"  {g:<12} {v['fotos']:>3} fotos   A {v['acertos_A']:>3} ({v['acuracia_A'] * 100:5.1f}%)   "
                  f"B {v['acertos_B']:>3} ({v['acuracia_B'] * 100:5.1f}%)")
    per_coin = results["por_exemplar"]
    results["media_por_exemplar"] = {
        "A": float(np.mean([v["acuracia_A"] for v in per_coin.values()])),
        "B": float(np.mean([v["acuracia_B"] for v in per_coin.values()])),
        "exemplares_todas_certas_A": sum(v["acertos_A"] == v["fotos"] for v in per_coin.values()),
        "exemplares_todas_certas_B": sum(v["acertos_B"] == v["fotos"] for v in per_coin.values())}
    m = results["media_por_exemplar"]
    print(f"\nMédia da acurácia por moeda: A {m['A'] * 100:.1f}% | B {m['B'] * 100:.1f}%  "
          f"(moedas com todas as fotos certas: A {m['exemplares_todas_certas_A']}, "
          f"B {m['exemplares_todas_certas_B']} de {len(per_coin)})")

    save_json(results, out_dir / "resultados_avaliacao_final.json")
    with open(out_dir / "fotos_avaliacao_final.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["arquivo", "exemplar", "classe", "face", "sessao", "luz", "fundo", "angulo", "acabamento",
                         "resposta_A", "acertou_A", "resposta_B", "acertou_B", "circulo_encontrado"]
                        + [f"A_{n}" for n in args.nomes] + [f"B_{n}" for n in args.nomes]
                        + [f"A_prob_{short(c)}" for c in class_names] + [f"B_prob_{short(c)}" for c in class_names])
        mean_a, mean_b = probs["A"].mean(axis=0), probs["B"].mean(axis=0)
        for i, it in enumerate(items):
            writer.writerow([it["arquivo"], it["exemplar"], it["classe"], it["face"], it["sessao"], it["luz"],
                             it["fundo"], it["angulo"], it["acabamento"],
                             short(class_names[preds["A"][i]]), "sim" if ok_a[i] else "NAO",
                             short(class_names[preds["B"][i]]), "sim" if ok_b[i] else "NAO",
                             "sim" if found[i] else "NAO"]
                            + [short(class_names[int(p[i].argmax())]) for p in probs["A"]]
                            + [short(class_names[int(p[i].argmax())]) for p in probs["B"]]
                            + [f"{x * 100:.1f}" for x in mean_a[i]] + [f"{x * 100:.1f}" for x in mean_b[i]])
    print(f"\nSalvo em: {out_dir}")
    print("  resultados_avaliacao_final.json, fotos_avaliacao_final.csv, matriz_confusao_*.png, recortes\\")


if __name__ == "__main__":
    main()
