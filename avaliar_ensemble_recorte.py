import argparse
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from config import DATASET_DIR, IMAGE_SIZE
from io_utils import save_json
from metrics import build_confusion_matrix, compute_metrics, plot_confusion_matrix
from prever_moeda_recortada import isolate_coin, load_rgb

# Igual ao avaliar_ensemble.py (mesmos modelos, mesma média das probabilidades), mas cada foto passa antes pelo
# pré-processamento do prever_moeda_recortada.py: aplica a rotação do celular (EXIF), encontra o círculo da moeda,
# recorta um quadrado em volta e pinta de branco o que está fora do círculo. Serve para comparar, nos MESMOS
# conjuntos, o pipeline original com o pipeline com recorte. Não treina nada e não altera nenhum modelo.
# Salva as imagens recortadas (para conferir a detecção) e conta em quantas fotos o círculo não foi encontrado.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia o ensemble com o pré-processamento de recorte da moeda.")
    parser.add_argument("--modelos", nargs="+", required=True, help="Arquivos .keras (ex.: as 3 sementes)")
    parser.add_argument("--nomes", nargs="+", required=True, help="Um nome curto por modelo (ex.: s42 s1 s2)")
    parser.add_argument("--pasta", required=True, help="Pasta com uma subpasta por classe")
    parser.add_argument("--saida", required=True, help="Pasta onde salvar os resultados")
    parser.add_argument("--fundo", type=int, default=255, help="Cor do fundo, 0-255 (padrão: 255 = branco)")
    return parser.parse_args()


def short(name: str) -> str:
    return name.split()[0]


def main() -> None:
    args = parse_args()
    if len(args.modelos) != len(args.nomes):
        raise SystemExit("Passe um nome para cada modelo (--nomes).")
    out_dir = Path(args.saida)
    if out_dir.exists():
        raise SystemExit(f"A pasta {out_dir} já existe. Use outra --saida.")
    class_names = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())
    folder = Path(args.pasta)

    paths, labels = [], []
    for class_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        if class_dir.name not in class_names:
            raise SystemExit(f"Pasta '{class_dir.name}' não é uma das classes do modelo: {class_names}")
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(p)
                labels.append(class_names.index(class_dir.name))
    if not paths:
        raise SystemExit(f"Nenhuma foto encontrada nas subpastas de {folder}")
    labels = np.array(labels)

    out_dir.mkdir(parents=True)
    crops_dir = out_dir / "recortes"
    print(f"Fotos: {len(paths)}. Recortando...", flush=True)
    tensors, found = [], []
    for i, path in enumerate(paths):
        coin, ok = isolate_coin(load_rgb(str(path)), args.fundo)
        found.append(ok)
        target = crops_dir / path.parent.name / f"{path.stem}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(coin).save(target, quality=90)
        tensors.append(tf.image.resize(tf.constant(coin[None], dtype=tf.float32), IMAGE_SIZE))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(paths)}", flush=True)
    found = np.array(found)
    print(f"Círculo encontrado em {found.sum()} de {len(paths)} fotos "
          f"({len(paths) - found.sum()} usaram o quadrado central, sem pintar o fundo)", flush=True)

    probs = []  # [modelo, foto, classe]
    for model_path, nome in zip(args.modelos, args.nomes):
        print(f"Avaliando {nome} ...", flush=True)
        model = tf.keras.models.load_model(model_path)
        p = [model(tf.concat(tensors[s:s + 32], axis=0), training=False).numpy() for s in range(0, len(paths), 32)]
        probs.append(np.concatenate(p))
    probs = np.stack(probs)
    mean_probs = probs.mean(axis=0)

    results = {"modelos": dict(zip(args.nomes, args.modelos)), "pasta": str(folder),
               "preprocessamento": "recorte da moeda (prever_moeda_recortada.isolate_coin), fundo = "
                                   f"{args.fundo}", "circulo_encontrado": int(found.sum()),
               "total_imagens": len(paths), "individual": {}}
    print(f"\n{'Modelo':<12}{'Acurácia':>10}{'Acur. balanceada':>18}")
    accs = []
    for nome, p in zip(args.nomes, probs):
        m = compute_metrics(build_confusion_matrix(labels, p.argmax(axis=1), len(class_names)), class_names)
        results["individual"][nome] = m
        accs.append(m["acuracia"])
        print(f"{nome:<12}{m['acuracia'] * 100:>9.1f}%{m['revocacao_macro'] * 100:>17.1f}%")
    sd = np.std(accs, ddof=1) * 100 if len(accs) > 1 else 0.0
    print(f"{'média':<12}{np.mean(accs) * 100:>9.1f}%  (± {sd:.1f})")

    pred = mean_probs.argmax(axis=1)
    matrix = build_confusion_matrix(labels, pred, len(class_names))
    ens = compute_metrics(matrix, class_names)
    results["media_individual"] = {"acuracia": float(np.mean(accs)), "desvio": float(sd / 100)}
    results["ensemble"] = ens
    ok_found = (pred == labels)[found].mean() if found.any() else float("nan")
    ok_missing = (pred == labels)[~found].mean() if (~found).any() else float("nan")
    results["ensemble_por_deteccao"] = {"com_circulo": float(ok_found), "sem_circulo": float(ok_missing)}
    print(f"{'ENSEMBLE':<12}{ens['acuracia'] * 100:>9.1f}%{ens['revocacao_macro'] * 100:>17.1f}%")
    print(f"  ensemble nas fotos com círculo encontrado: {ok_found * 100:.1f}% | sem círculo: {ok_missing * 100:.1f}%")
    print("Matriz do ensemble (linhas = moeda certa, colunas = resposta):")
    for name, row in zip(class_names, matrix):
        print(f"  {short(name):<12}{row}")

    save_json(results, out_dir / "resultados_ensemble_recorte.json")
    plot_confusion_matrix(matrix, class_names, out_dir / "matriz_confusao_ensemble_recorte.png")
    with open(out_dir / "fotos_ensemble_recorte.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["arquivo", "moeda_certa", "ensemble", "acertou", "circulo_encontrado"]
                        + [f"{n}_resposta" for n in args.nomes]
                        + [f"ens_prob_{short(c)}" for c in class_names])
        for i, path in enumerate(paths):
            writer.writerow([path.name, short(class_names[labels[i]]), short(class_names[pred[i]]),
                             "sim" if pred[i] == labels[i] else "NAO", "sim" if found[i] else "NAO"]
                            + [short(class_names[int(p[i].argmax())]) for p in probs]
                            + [f"{x * 100:.1f}" for x in mean_probs[i]])
    print(f"\nSalvo em: {out_dir}")
    print("  resultados_ensemble_recorte.json, fotos_ensemble_recorte.csv, recortes\\ (imagens que o modelo recebeu)")


if __name__ == "__main__":
    main()
