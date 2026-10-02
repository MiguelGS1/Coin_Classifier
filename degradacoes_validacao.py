import csv

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import ARTIFACTS_DIR, CLASS_NAMES_PATH, MODEL_PATH
from dataset_loader import build_datasets
from io_utils import load_json

OUTPUT_DIR = ARTIFACTS_DIR / "experiments" / "degradacoes"
VARIANTS_DIR = ARTIFACTS_DIR / "experiments" / "celular" / "modelos"
MODELS = {
    "atual": MODEL_PATH,
    "sem_aumento": VARIANTS_DIR / "sem_aumento.keras",
    "rotacao": VARIANTS_DIR / "rotacao.keras",
    "luz_cor": VARIANTS_DIR / "luz_cor.keras",
    "forte": VARIANTS_DIR / "forte.keras",
}


def desfoque(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)


def luz_amarelada(img, a):
    # Imagem em RGB: aumenta o vermelho e reduz o azul (como luz de lâmpada)
    out = img.astype(np.float32)
    out[..., 0] *= 1 + a
    out[..., 2] *= 1 - a
    return np.clip(out, 0, 255).astype(np.uint8)


def escurecimento(img, fator):
    return np.clip(img.astype(np.float32) * fator, 0, 255).astype(np.uint8)


def baixa_resolucao(img, lado):
    small = cv2.resize(img, (lado, lado), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, img.shape[1::-1], interpolation=cv2.INTER_LINEAR)


# Fatores e níveis definidos antes de ver os resultados (nível 0 = imagem original, avaliada uma vez)
FACTORS = {
    "desfoque": (desfoque, [0.5, 1, 1.5, 2, 3, 4], "σ do desfoque gaussiano (pixels)"),
    "luz_amarelada": (luz_amarelada, [0.1, 0.2, 0.3, 0.4, 0.5], "intensidade (vermelho +a, azul −a)"),
    "escurecimento": (escurecimento, [0.8, 0.6, 0.4, 0.3, 0.2], "fator de brilho"),
    "baixa_resolucao": (baixa_resolucao, [112, 64, 48, 32, 24], "lado da imagem reduzida (pixels)"),
}


def load_validation():
    _, val_ds, class_names = build_datasets()
    images, labels = [], []
    for x, y in val_ds:
        images.append(np.clip(x.numpy(), 0, 255).astype(np.uint8))
        labels.append(np.argmax(y.numpy(), axis=1))
    return np.concatenate(images), np.concatenate(labels), list(class_names)


def evaluate(model, images, labels, n_classes):
    preds = []
    for start in range(0, len(images), 32):
        batch = tf.constant(images[start:start + 32], dtype=tf.float32)
        preds.append(np.argmax(model(batch, training=False).numpy(), axis=1))
    preds = np.concatenate(preds)
    recalls = [float((preds[labels == c] == c).mean()) for c in range(n_classes)]
    return float((preds == labels).mean()), recalls


def plot_curves(rows, path):
    # Eixo x: passo da degradação (0 = original); os rótulos mostram o valor de cada nível
    fig, axes = plt.subplots(1, len(FACTORS), figsize=(4.2 * len(FACTORS), 3.6))
    for ax, (factor, (_, levels, xlabel)) in zip(axes, FACTORS.items()):
        for model_name in dict.fromkeys(r["modelo"] for r in rows):
            pts = [r for r in rows if r["modelo"] == model_name and r["fator"] in ("original", factor)]
            ax.plot(range(len(pts)), [r["acuracia"] * 100 for r in pts], marker="o", label=model_name)
        ax.set_xticks(range(len(levels) + 1))
        ax.set_xticklabels(["orig."] + [str(l) for l in levels], fontsize=8)
        ax.set_title(factor.replace("_", " "))
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Acurácia na validação (%)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_examples(image, path):
    fig, axes = plt.subplots(len(FACTORS), 7, figsize=(13, 2.1 * len(FACTORS)))
    for row, (factor, (fn, levels, _)) in enumerate(FACTORS.items()):
        imgs = [("original", image)] + [(str(l), fn(image, l)) for l in levels]
        for col in range(7):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(imgs):
                ax.imshow(imgs[col][1])
                ax.set_title(f"{factor.replace('_', ' ')} {imgs[col][0]}" if col else "original", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images, labels, class_names = load_validation()
    if class_names != list(load_json(CLASS_NAMES_PATH)):
        raise ValueError("As classes do dataset diferem das do modelo final.")
    n = len(class_names)

    # As imagens degradadas são calculadas uma vez e usadas em todos os modelos
    versions = [("original", 0, images)]
    for factor, (fn, levels, _) in FACTORS.items():
        for level in levels:
            versions.append((factor, level, np.stack([fn(img, level) for img in images])))

    rows = []
    for model_name, path in MODELS.items():
        if not path.exists():
            print(f"Pulando '{model_name}': modelo não encontrado em {path}")
            continue
        print(f"Avaliando o modelo '{model_name}'...")
        model = tf.keras.models.load_model(path)
        for factor, level, imgs in versions:
            acc, recalls = evaluate(model, imgs, labels, n)
            rows.append({"modelo": model_name, "fator": factor, "nivel": level, "acuracia": acc,
                         **{class_names[c].split()[0]: recalls[c] for c in range(n)}})

    with open(OUTPUT_DIR / "degradacoes.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        short = [c.split()[0] for c in class_names]
        writer.writerow(["modelo", "fator", "nivel", "acuracia"] + short)
        pct = lambda v: f"{v * 100:.1f}".replace(".", ",")
        for r in rows:
            writer.writerow([r["modelo"], r["fator"], str(r["nivel"]).replace(".", ","), pct(r["acuracia"])]
                            + [pct(r[s]) for s in short])
    plot_curves(rows, OUTPUT_DIR / "curvas_degradacoes.png")
    plot_examples(images[0], OUTPUT_DIR / "exemplo_degradacoes.png")

    print("\nAcurácia na validação (%) por fator e nível")
    for factor, (_, levels, xlabel) in FACTORS.items():
        print(f"\n{factor} ({xlabel})")
        print(f"  {'modelo':<12}{'original':>9}" + "".join(f"{str(l):>8}" for l in levels))
        for model_name in dict.fromkeys(r["modelo"] for r in rows):
            vals = [r for r in rows if r["modelo"] == model_name and r["fator"] in ("original", factor)]
            print(f"  {model_name:<12}" + f"{vals[0]['acuracia'] * 100:>9.1f}"
                  + "".join(f"{v['acuracia'] * 100:>8.1f}" for v in vals[1:]))
    print(f"\nArquivos salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
