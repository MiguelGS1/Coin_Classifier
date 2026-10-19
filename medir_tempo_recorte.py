import argparse
import platform
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import IMAGE_SIZE
from dataset_loader import build_single_image_tensor
from prever_moeda_recortada import isolate_coin, load_rgb

# Mede o tempo POR IMAGEM das duas configurações, uma foto de cada vez (como num aplicativo):
#   A = ensemble sem recorte: ler a foto + preparar + 3 redes + média (igual ao medir_tempo.py);
#   B = ensemble com recorte: ler a foto + recortar a moeda (prever_moeda_recortada) + redimensionar + 3 redes + média.
# Também mede separadamente só o recorte, para mostrar quanto ele acrescenta. As primeiras fotos são aquecimento e
# não entram na conta. Só mede; não altera nada.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mede o tempo por imagem do ensemble sem e com recorte.")
    parser.add_argument("--modelos", nargs="+", required=True, help="Arquivos .keras (as 3 sementes)")
    parser.add_argument("--pasta", required=True, help="Pasta com fotos (pode ter subpastas)")
    parser.add_argument("--fotos", type=int, default=100, help="Quantas fotos medir (padrão: 100)")
    parser.add_argument("--aquecimento", type=int, default=10, help="Fotos de aquecimento, fora da conta")
    parser.add_argument("--fundo", type=int, default=255, help="Cor do fundo do recorte (padrão: 255 = branco)")
    return parser.parse_args()


def summary(name: str, values: list) -> None:
    ms = np.array(values) * 1000
    print(f"  {name:<36}média {ms.mean():7.1f} ms   mediana {np.median(ms):7.1f} ms   "
          f"p95 {np.percentile(ms, 95):7.1f} ms")


def main() -> None:
    args = parse_args()
    paths = sorted(p for p in Path(args.pasta).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(paths) < args.aquecimento + 1:
        raise SystemExit("Fotos de menos na pasta.")
    rng = np.random.default_rng(0)  # mesma amostra de fotos do medir_tempo.py
    paths = [paths[i] for i in rng.permutation(len(paths))[: args.aquecimento + args.fotos]]

    print(f"Computador: {platform.processor() or platform.machine()} | {platform.system()} | "
          f"GPU: {tf.config.list_physical_devices('GPU') or 'nenhuma'}", flush=True)
    models = [tf.keras.models.load_model(m) for m in args.modelos]
    print(f"Medindo {len(paths)} fotos ({args.aquecimento} de aquecimento)...", flush=True)

    times = {"a_total": [], "b_recorte": [], "b_rede": [], "b_total": []}
    for i, path in enumerate(paths):
        # A: sem recorte
        t0 = time.perf_counter()
        image = build_single_image_tensor(str(path))
        np.mean([m(image, training=False).numpy()[0] for m in models], axis=0).argmax()
        t1 = time.perf_counter()
        # B: com recorte
        t2 = time.perf_counter()
        coin, _ = isolate_coin(load_rgb(str(path)), args.fundo)
        image = tf.image.resize(tf.constant(coin[None], dtype=tf.float32), IMAGE_SIZE)
        t3 = time.perf_counter()
        np.mean([m(image, training=False).numpy()[0] for m in models], axis=0).argmax()
        t4 = time.perf_counter()
        if i == 0 or (i + 1) % 10 == 0:
            print(f"  foto {i + 1}/{len(paths)}: A {(t1 - t0) * 1000:.0f} ms, B {(t4 - t2) * 1000:.0f} ms",
                  flush=True)
        if i >= args.aquecimento:
            times["a_total"].append(t1 - t0)
            times["b_recorte"].append(t3 - t2)
            times["b_rede"].append(t4 - t3)
            times["b_total"].append(t4 - t2)

    print(f"\nTempo por imagem ({args.fotos} fotos, uma de cada vez):")
    summary("A sem recorte - total", times["a_total"])
    summary("B com recorte - só ler + recortar", times["b_recorte"])
    summary(f"B com recorte - só as {len(models)} redes", times["b_rede"])
    summary("B com recorte - total", times["b_total"])
    print(f"\nO recorte acrescenta em média {(np.mean(times['b_total']) - np.mean(times['a_total'])) * 1000:.1f} ms "
          f"por imagem ({np.mean(times['b_total']) / np.mean(times['a_total']):.2f}x o tempo de A).")


if __name__ == "__main__":
    main()
