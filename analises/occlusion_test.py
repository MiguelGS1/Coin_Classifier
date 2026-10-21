import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import ARTIFACTS_DIR, CLASS_NAMES_PATH, IMAGE_SIZE, MODEL_PATH
from dataset_loader import build_datasets
from io_utils import load_json, save_json

OUTPUT_DIR = ARTIFACTS_DIR / "experiments" / "oclusao"

# A moeda é aproximada por um círculo no centro da imagem, com raio de 45% do lado.
# O "centro" é o disco interno (60% desse raio) e a "borda" é o anel entre 60% e 100%.
COIN_RADIUS = 0.45
CENTER_FRACTION = 0.60
GRAY = 128.0  # valor usado para cobrir as regiões (cinza médio, na escala 0-255)

TESTS = {
    "original": "Imagem original",
    "sem_centro": "Centro coberto",
    "sem_borda": "Borda coberta",
    "sem_fundo": "Fundo coberto",
    "tons_de_cinza": "Tons de cinza",
}


def region_masks() -> dict:
    height, width = IMAGE_SIZE
    yy, xx = np.mgrid[0:height, 0:width]
    dist = np.hypot(yy - (height - 1) / 2, xx - (width - 1) / 2) / min(height, width)
    center = dist <= COIN_RADIUS * CENTER_FRACTION
    coin = dist <= COIN_RADIUS
    return {
        "sem_centro": center,
        "sem_borda": coin & ~center,
        "sem_fundo": ~coin,
    }


def transform(images: tf.Tensor, test: str, masks: dict) -> tf.Tensor:
    if test == "original":
        return images
    if test == "tons_de_cinza":
        return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(images))
    mask = tf.constant(masks[test][None, :, :, None], dtype=images.dtype)
    return images * (1 - mask) + GRAY * mask


def evaluate(model, val_ds, test: str, masks: dict):
    y_true, y_pred, true_conf = [], [], []
    for images, labels in val_ds:
        probs = model(transform(images, test, masks), training=False).numpy()
        true_idx = np.argmax(labels.numpy(), axis=1)
        y_true.append(true_idx)
        y_pred.append(np.argmax(probs, axis=1))
        true_conf.append(probs[np.arange(len(probs)), true_idx])
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(true_conf)


def plot_masks_example(val_ds, masks: dict, path) -> None:
    # Mostra, numa imagem de validação, como fica cada teste
    images, _ = next(iter(val_ds))
    image = images[:1]
    fig, axes = plt.subplots(1, len(TESTS), figsize=(2.4 * len(TESTS), 2.8))
    for ax, (test, label) in zip(axes, TESTS.items()):
        ax.imshow(np.clip(transform(image, test, masks)[0].numpy() / 255.0, 0, 1))
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def short(name: str) -> str:
    return name.split(",")[0]


def latex_percent(value: float) -> str:
    return f"{value * 100:.1f}".replace(".", ",") + "\\%"


def build_latex_table(results: dict, class_names: list) -> str:
    cols = "l" + "r" * (len(class_names) + 1)
    header = "Teste & " + " & ".join(short(n) for n in class_names) + " & Geral \\\\"
    lines = [f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}", "\\toprule", header, "\\midrule"]
    for test, label in TESTS.items():
        r = results[test]
        values = [latex_percent(r["acuracia_por_classe"][n]) for n in class_names]
        lines.append(f"{label} & " + " & ".join(values) + f" & {latex_percent(r['acuracia'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, val_ds, class_names = build_datasets()
    saved_class_names = load_json(CLASS_NAMES_PATH)
    if list(saved_class_names) != list(class_names):
        raise ValueError("As classes do dataset atual diferem das classes do modelo treinado.")

    model = tf.keras.models.load_model(MODEL_PATH)
    masks = region_masks()

    results = {}
    for test, label in TESTS.items():
        print(f"Testando: {label}...")
        y_true, y_pred, true_conf = evaluate(model, val_ds, test, masks)
        per_class = {}
        predicted_as = {}
        for i, name in enumerate(class_names):
            sel = y_true == i
            per_class[name] = float((y_pred[sel] == i).mean())
            predicted_as[name] = {class_names[j]: int((y_pred[sel] == j).sum()) for j in range(len(class_names))}
        total_images = len(y_true)
        results[test] = {
            "acuracia": float((y_true == y_pred).mean()),
            "acuracia_por_classe": per_class,
            "confianca_media_classe_correta": float(true_conf.mean()),
            "previsoes_por_classe_verdadeira": predicted_as,
        }

    save_json({"raio_moeda": COIN_RADIUS, "fracao_centro": CENTER_FRACTION, "resultados": results},
              OUTPUT_DIR / "resultados_oclusao.json")
    (OUTPUT_DIR / "tabela_oclusao.tex").write_text(build_latex_table(results, class_names), encoding="utf-8")
    plot_masks_example(val_ds, masks, OUTPUT_DIR / "exemplo_testes.png")

    print(f"\nTeste de oclusão e cor - acurácia na validação ({total_images} imagens)")
    header = f"{'Teste':<18}" + "".join(f"{short(n):>20}" for n in class_names) + f"{'Geral':>10}"
    print(header)
    for test, label in TESTS.items():
        r = results[test]
        row = f"{label:<18}" + "".join(f"{r['acuracia_por_classe'][n] * 100:>19.1f}%" for n in class_names)
        print(row + f"{r['acuracia'] * 100:>9.1f}%")

    print("\nPara onde foram as previsões em cada teste (linhas = classe verdadeira):")
    for test, label in TESTS.items():
        print(f"  {label}:")
        for name, counts in results[test]["previsoes_por_classe_verdadeira"].items():
            text = ", ".join(f"{short(k)} {v}" for k, v in counts.items())
            print(f"    {short(name):<20} -> {text}")
    print(f"\nArquivos salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
