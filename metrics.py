import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import ARTIFACTS_DIR, CLASS_NAMES_PATH, MODEL_PATH, TRAINING_HISTORY_PATH
from dataset_loader import build_datasets
from io_utils import load_json, save_json

OUTPUT_DIR = ARTIFACTS_DIR / "experiments" / "exp_a"


def collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset):
    y_true, y_pred = [], []
    for batch_images, batch_labels in dataset:
        probs = model.predict(batch_images, verbose=0)
        y_true.append(np.argmax(batch_labels.numpy(), axis=1))
        y_pred.append(np.argmax(probs, axis=1))
    return np.concatenate(y_true), np.concatenate(y_pred)


def build_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    # Linhas = classe verdadeira, colunas = classe prevista
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    )


def compute_metrics(matrix: np.ndarray, class_names: list[str]) -> dict:
    true_positives = np.diag(matrix).astype(float)
    predicted_per_class = matrix.sum(axis=0).astype(float)
    actual_per_class = matrix.sum(axis=1).astype(float)

    precision = safe_divide(true_positives, predicted_per_class)
    recall = safe_divide(true_positives, actual_per_class)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    per_class = {
        name: {
            "precisao": float(precision[i]),
            "revocacao": float(recall[i]),
            "f1": float(f1[i]),
            "suporte": int(actual_per_class[i]),
        }
        for i, name in enumerate(class_names)
    }

    return {
        "acuracia": float(true_positives.sum() / matrix.sum()),
        "precisao_macro": float(precision.mean()),
        "revocacao_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "total_imagens": int(matrix.sum()),
        "por_classe": per_class,
        "classes": class_names,
        "matriz_confusao": matrix.tolist(),
    }


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], path) -> None:
    # Porcentagem por linha: de cada classe verdadeira, para onde foram as previsões
    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = safe_divide(matrix.astype(float), row_totals.astype(float)) * 100

    n = len(class_names)
    size = max(4.5, 1.1 * n + 2)
    labels = ["\n".join(textwrap.wrap(name, 16)) for name in class_names]

    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="%")

    for i in range(n):
        for j in range(n):
            color = "white" if percentages[i, j] > 60 else "black"
            ax.text(
                j, i, f"{percentages[i, j]:.1f}%".replace(".", ",") + f"\n({matrix[i, j]})",
                ha="center", va="center", color=color, fontsize=9,
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe verdadeira")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_training_curves(history: dict, path) -> int:
    epochs = np.arange(1, len(history["loss"]) + 1)
    best_epoch = int(np.argmin(history["val_loss"])) + 1

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(9, 3.5))

    ax_loss.plot(epochs, history["loss"], marker="o", markersize=3, label="Treinamento")
    ax_loss.plot(epochs, history["val_loss"], marker="o", markersize=3, label="Validação")
    ax_loss.set_title("Perda")

    ax_acc.plot(epochs, history["accuracy"], marker="o", markersize=3, label="Treinamento")
    ax_acc.plot(epochs, history["val_accuracy"], marker="o", markersize=3, label="Validação")
    ax_acc.set_title("Acurácia")

    for ax in (ax_loss, ax_acc):
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Época")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return best_epoch


def latex_number(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def latex_escape(text: str) -> str:
    for char in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(char, "\\" + char)
    return text


def build_latex_table(metrics: dict) -> str:
    lines = [
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "Classe & Precisão & Revocação & F1 & Suporte \\\\",
        "\\midrule",
    ]
    for name, values in metrics["por_classe"].items():
        lines.append(
            f"{latex_escape(name)} & {latex_number(values['precisao'])} & "
            f"{latex_number(values['revocacao'])} & {latex_number(values['f1'])} & "
            f"{values['suporte']} \\\\"
        )
    lines += [
        "\\midrule",
        f"Média macro & {latex_number(metrics['precisao_macro'])} & "
        f"{latex_number(metrics['revocacao_macro'])} & {latex_number(metrics['f1_macro'])} & "
        f"{metrics['total_imagens']} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _, val_ds, class_names = build_datasets()
    saved_class_names = load_json(CLASS_NAMES_PATH)
    if list(saved_class_names) != list(class_names):
        raise ValueError(
            "As classes do dataset atual diferem das classes do modelo treinado. "
            "Rode python train.py novamente antes de calcular as métricas."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    y_true, y_pred = collect_predictions(model, val_ds)

    matrix = build_confusion_matrix(y_true, y_pred, len(class_names))
    metrics = compute_metrics(matrix, class_names)

    history = load_json(TRAINING_HISTORY_PATH)
    best_epoch = plot_training_curves(history, OUTPUT_DIR / "curvas_treinamento.png")
    metrics["epocas_executadas"] = len(history["loss"])
    metrics["melhor_epoca"] = best_epoch

    plot_confusion_matrix(matrix, class_names, OUTPUT_DIR / "matriz_confusao.png")
    save_json(metrics, OUTPUT_DIR / "metricas.json")
    (OUTPUT_DIR / "tabela_metricas.tex").write_text(build_latex_table(metrics), encoding="utf-8")

    print("\nExperimento A - métricas no conjunto de validação")
    print(f"Imagens avaliadas: {metrics['total_imagens']}")
    print(f"Épocas executadas: {metrics['epocas_executadas']} (melhor época: {best_epoch})\n")
    print(f"{'Classe':<35}{'Precisão':>10}{'Revocação':>11}{'F1':>8}{'Suporte':>9}")
    for name, values in metrics["por_classe"].items():
        print(
            f"{name[:34]:<35}{values['precisao']:>10.2f}{values['revocacao']:>11.2f}"
            f"{values['f1']:>8.2f}{values['suporte']:>9}"
        )
    print(
        f"{'Média macro':<35}{metrics['precisao_macro']:>10.2f}"
        f"{metrics['revocacao_macro']:>11.2f}{metrics['f1_macro']:>8.2f}"
        f"{metrics['total_imagens']:>9}"
    )
    print(f"\nAcurácia: {metrics['acuracia'] * 100:.2f}%")
    print(f"\nArquivos salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()