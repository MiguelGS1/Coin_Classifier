import argparse
import platform
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset_loader import build_single_image_tensor

# Mede o tempo de reconhecimento POR IMAGEM, uma foto de cada vez (como num aplicativo): com um modelo só e com o
# ensemble (todos os modelos, um depois do outro, e a média das probabilidades). Mede duas coisas:
#   - só a rede (a foto já carregada);
#   - o total (ler o arquivo + preparar a imagem + rede + média).
# As primeiras fotos servem de "aquecimento" e não entram na conta. Rode no computador onde o reconhecimento vai rodar.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mede o tempo por imagem do modelo individual e do ensemble.")
    parser.add_argument("--modelos", nargs="+", required=True, help="Arquivos .keras; o primeiro é o 'individual'")
    parser.add_argument("--pasta", required=True, help="Pasta com fotos (pode ter subpastas)")
    parser.add_argument("--fotos", type=int, default=100, help="Quantas fotos medir (padrão: 100)")
    parser.add_argument("--aquecimento", type=int, default=10, help="Fotos de aquecimento, fora da conta")
    return parser.parse_args()


def summary(name: str, values: list) -> None:
    ms = np.array(values) * 1000
    print(f"  {name:<32}média {ms.mean():7.1f} ms   mediana {np.median(ms):7.1f} ms   "
          f"p95 {np.percentile(ms, 95):7.1f} ms")


def main() -> None:
    args = parse_args()
    paths = sorted(p for p in Path(args.pasta).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(paths) < args.aquecimento + 1:
        raise SystemExit("Fotos de menos na pasta.")
    rng = np.random.default_rng(0)
    paths = [paths[i] for i in rng.permutation(len(paths))[: args.aquecimento + args.fotos]]

    print(f"Computador: {platform.processor() or platform.machine()} | {platform.system()} | "
          f"GPU: {tf.config.list_physical_devices('GPU') or 'nenhuma'}")
    t0 = time.perf_counter()
    models = [tf.keras.models.load_model(m) for m in args.modelos]
    print(f"Carregar os {len(models)} modelos (só uma vez, ao abrir o programa): {time.perf_counter() - t0:.1f} s")

    times = {"ind_rede": [], "ind_total": [], "ens_rede": [], "ens_total": []}
    for i, path in enumerate(paths):
        # Individual
        t0 = time.perf_counter()
        image = build_single_image_tensor(str(path))
        t1 = time.perf_counter()
        models[0](image, training=False).numpy()
        t2 = time.perf_counter()
        # Ensemble (imagem carregada de novo, para medir o total do mesmo jeito)
        t3 = time.perf_counter()
        image = build_single_image_tensor(str(path))
        t4 = time.perf_counter()
        np.mean([m(image, training=False).numpy()[0] for m in models], axis=0).argmax()
        t5 = time.perf_counter()
        if i >= args.aquecimento:
            times["ind_rede"].append(t2 - t1)
            times["ind_total"].append(t2 - t0)
            times["ens_rede"].append(t5 - t4)
            times["ens_total"].append(t5 - t3)

    print(f"\nTempo por imagem ({args.fotos} fotos, uma de cada vez):")
    summary("Individual - só a rede", times["ind_rede"])
    summary(f"Ensemble ({len(models)}) - só a rede", times["ens_rede"])
    summary("Individual - total", times["ind_total"])
    summary(f"Ensemble ({len(models)}) - total", times["ens_total"])
    ratio = np.mean(times["ens_total"]) / np.mean(times["ind_total"])
    print(f"\nO ensemble leva {ratio:.1f}x o tempo do modelo individual (total por imagem).")


if __name__ == "__main__":
    main()
