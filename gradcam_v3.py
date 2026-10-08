import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import ARTIFACTS_DIR, BASE_DIR, DATASET_DIR, IMAGE_SIZE, SEED
from dataset_loader import build_single_image_tensor
from io_utils import save_json

# Grad-CAM de um ou mais modelos nas MESMAS fotos do teste (séries nunca vistas), lado a lado.
# Também mede, em todas as fotos do teste, quanto do mapa de calor cai DENTRO da moeda (círculo central de
# raio 0,45 do lado, o mesmo do teste de oclusão). Não treina nada; só lê os modelos e salva figuras.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
COIN_RADIUS = 0.45


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM de modelos no teste da divisão v2.")
    parser.add_argument("--modelos", nargs="+", required=True, help="Arquivos .keras (um ou mais)")
    parser.add_argument("--nomes", nargs="+", required=True, help="Um nome curto por modelo (ex.: baseline lpr)")
    parser.add_argument("--pasta", default=str(BASE_DIR / "dataset" / "split_series_v2" / "teste"))
    parser.add_argument("--saida", default=str(ARTIFACTS_DIR / "experiments" / "gradcam_v2"))
    parser.add_argument("--por_classe", type=int, default=6, help="Fotos por moeda nas figuras")
    return parser.parse_args()


def make_grad_model(model: tf.keras.Model) -> tf.keras.Model:
    last_conv = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
    return tf.keras.Model(model.inputs, [last_conv.output, model.output])


def grad_cam(grad_model: tf.keras.Model, image: tf.Tensor):
    with tf.GradientTape() as tape:
        conv, probs = grad_model(image, training=False)
        pred = int(tf.argmax(probs[0]))
        score = probs[:, pred]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(conv * weights, axis=-1))[0]
    cam = tf.image.resize(cam[..., None], IMAGE_SIZE)[..., 0].numpy()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam, pred, float(probs[0, pred])


def coin_mask() -> np.ndarray:
    h, w = IMAGE_SIZE
    yy, xx = np.mgrid[0:h, 0:w]
    return (yy - h / 2) ** 2 + (xx - w / 2) ** 2 <= (COIN_RADIUS * min(h, w)) ** 2


def main() -> None:
    args = parse_args()
    if len(args.modelos) != len(args.nomes):
        raise SystemExit("Passe um nome para cada modelo (--nomes).")
    out_dir = Path(args.saida)
    if out_dir.exists():
        raise SystemExit(f"A pasta {out_dir} já existe. Use outra --saida.")
    class_names = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())
    folder = Path(args.pasta)
    grad_models = []
    for m in args.modelos:  # avisa a cada modelo carregado, para saber onde parou se der erro
        print(f"Carregando {m} ...", flush=True)
        grad_models.append(make_grad_model(tf.keras.models.load_model(m)))
    print("Modelos carregados. Calculando o Grad-CAM (mostra o progresso a cada 25 fotos)...", flush=True)
    start = time.time()
    mask = coin_mask()
    rng = np.random.default_rng(SEED)
    out_dir.mkdir(parents=True)

    summary = {nome: {} for nome in args.nomes}
    for true, cls in enumerate(class_names):
        paths = sorted(p for p in (folder / cls).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        shown = set(rng.choice(len(paths), size=min(args.por_classe, len(paths)), replace=False).tolist())
        inside = {nome: [] for nome in args.nomes}
        figure_rows = []
        for i, path in enumerate(paths):
            if i % 25 == 0:
                print(f"  {cls.split()[0]}: foto {i + 1} de {len(paths)} ({(time.time() - start) / 60:.1f} min)", flush=True)
            image = build_single_image_tensor(str(path))
            row = []
            for nome, grad_model in zip(args.nomes, grad_models):
                cam, pred, conf = grad_cam(grad_model, image)
                inside[nome].append(cam[mask].sum() / max(cam.sum(), 1e-8))
                row.append((cam, pred, conf))
            if i in shown:
                figure_rows.append((path, image[0].numpy().astype("uint8"), row))
        for nome in args.nomes:
            summary[nome][cls] = float(np.mean(inside[nome]))

        n_cols = 1 + len(args.nomes)
        fig, axes = plt.subplots(len(figure_rows), n_cols, figsize=(3 * n_cols, 3 * len(figure_rows)), squeeze=False)
        for r, (path, img, row) in enumerate(figure_rows):
            axes[r, 0].imshow(img)
            axes[r, 0].set_title(path.name[:28], fontsize=7)
            for c, (nome, (cam, pred, conf)) in enumerate(zip(args.nomes, row), start=1):
                axes[r, c].imshow(img)
                axes[r, c].imshow(cam, cmap="jet", alpha=0.45)
                axes[r, c].set_title(f"{nome}: {class_names[pred].split()[0]} ({conf:.2f})", fontsize=8,
                                     color="green" if pred == true else "red")
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(f"{cls} – teste (séries nunca vistas)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"gradcam_{cls.split()[0]}.png", dpi=150)
        plt.close(fig)

    save_json({"modelos": dict(zip(args.nomes, args.modelos)), "pasta": str(folder),
               "parte_do_mapa_dentro_da_moeda": summary}, out_dir / "resumo.json")
    print("\nParte do mapa de calor DENTRO da moeda (média no teste; o resto está no fundo/borda externa):")
    print(f"  {'modelo':<12}" + "".join(f"{c.split()[0]:>12}" for c in class_names))
    for nome in args.nomes:
        print(f"  {nome:<12}" + "".join(f"{summary[nome][c] * 100:>11.1f}%" for c in class_names))
    print(f"  (referência: um mapa espalhado por igual na imagem daria {mask.mean() * 100:.1f}%, que é a área do círculo)")
    print(f"\nFiguras e resumo em: {out_dir}")


if __name__ == "__main__":
    main()
