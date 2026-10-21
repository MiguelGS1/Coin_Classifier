import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from config import ARTIFACTS_DIR, DATASET_DIR
from dataset_loader import build_single_image_tensor
from gradcam_v2 import grad_cam, make_grad_model

# Grad-CAM de UMA foto (ou de algumas), com um ou mais modelos, lado a lado. Serve para fotos soltas fora do
# teste (ex.: uma foto baixada da internet). Usa as mesmas funções do gradcam_v2.py. Não treina nada.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM de uma ou mais fotos soltas.")
    parser.add_argument("--modelos", nargs="+", required=True, help="Arquivos .keras (um ou mais)")
    parser.add_argument("--nomes", nargs="+", required=True, help="Um nome curto por modelo")
    parser.add_argument("--imagem", nargs="+", required=True, help="Uma ou mais fotos")
    parser.add_argument("--saida", default=str(ARTIFACTS_DIR / "experiments" / "gradcam_imagens"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.modelos) != len(args.nomes):
        raise SystemExit("Passe um nome para cada modelo (--nomes).")
    class_names = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())
    grad_models = []
    for m in args.modelos:
        print(f"Carregando {m} ...", flush=True)
        grad_models.append(make_grad_model(tf.keras.models.load_model(m)))
    out_dir = Path(args.saida)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.imagem:
        image = build_single_image_tensor(path)
        img = image[0].numpy().astype("uint8")
        fig, axes = plt.subplots(1, 1 + len(grad_models), figsize=(3.2 * (1 + len(grad_models)), 3.6))
        axes[0].imshow(img)
        axes[0].set_title("foto", fontsize=9)
        print(f"\nFoto: {path}")
        for ax, nome, grad_model in zip(axes[1:], args.nomes, grad_models):
            cam, pred, conf = grad_cam(grad_model, image)
            ax.imshow(img)
            ax.imshow(cam, cmap="jet", alpha=0.45)
            ax.set_title(f"{nome}: {class_names[pred].split()[0]} ({conf:.2f})", fontsize=9)
            print(f"  {nome:<12} resposta: {class_names[pred]} ({conf * 100:.1f}%)")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        target = out_dir / f"gradcam_{Path(path).stem}.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        print(f"  figura salva em: {target}")


if __name__ == "__main__":
    main()
