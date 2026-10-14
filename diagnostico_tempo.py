import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset_loader import build_single_image_tensor

# Diagnóstico de lentidão: mede, com UM modelo e UMA foto, quanto leva cada forma de chamar a rede.
# Todas dão a MESMA resposta (confere no final); só muda a velocidade. Não altera nada.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnóstico de tempo de inferência.")
    parser.add_argument("--modelo", required=True)
    parser.add_argument("--pasta", required=True)
    parser.add_argument("--vezes", type=int, default=5)
    return parser.parse_args()


def timeit(name: str, fn, x, n: int) -> np.ndarray:
    t0 = time.perf_counter()
    out = fn(x)
    first = time.perf_counter() - t0
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn(x)
        times.append(time.perf_counter() - t0)
    print(f"  {name:<34} 1ª chamada {first * 1000:8.1f} ms | depois: mediana {np.median(times) * 1000:8.1f} ms",
          flush=True)
    return np.asarray(out)


def main() -> None:
    args = parse_args()
    import keras
    print(f"Python {platform.python_version()} | TensorFlow {tf.__version__} | Keras {keras.__version__} | "
          f"{platform.system()} | núcleos: {os.cpu_count()}", flush=True)
    print(f"Threads TF: intra={tf.config.threading.get_intra_op_parallelism_threads()} "
          f"inter={tf.config.threading.get_inter_op_parallelism_threads()} (0 = automático)", flush=True)
    path = next(p for p in sorted(Path(args.pasta).rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS)
    model = tf.keras.models.load_model(args.modelo)
    x = build_single_image_tensor(str(path))
    print(f"Foto: {path.name}\n", flush=True)

    t0 = time.perf_counter()
    for _ in range(args.vezes):
        build_single_image_tensor(str(path))
    print(f"  {'ler + preparar a foto':<34} mediana aprox. {(time.perf_counter() - t0) / args.vezes * 1000:8.1f} ms")

    a = timeit("model(x) (como nos scripts)", lambda v: model(v, training=False).numpy(), x, args.vezes)
    b = timeit("model.predict(x)", lambda v: model.predict(v, verbose=0), x, args.vezes)
    fast = tf.function(lambda v: model(v, training=False), reduce_retracing=True)
    c = timeit("tf.function(model)", lambda v: fast(v).numpy(), x, args.vezes)
    print(f"\nMesma resposta nas 3 formas? {np.allclose(a, b, atol=1e-4) and np.allclose(a, c, atol=1e-4)}  "
          f"(probabilidades: {np.round(a[0] * 100, 1)})")


if __name__ == "__main__":
    main()
