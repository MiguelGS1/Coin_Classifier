import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from config import ARTIFACTS_DIR, BASE_DIR, BATCH_SIZE, EARLY_STOPPING_PATIENCE, IMAGE_SIZE, LEARNING_RATE, SEED
from dataset_loader import build_datasets
from io_utils import save_json
from metrics import (
    build_confusion_matrix,
    collect_predictions,
    compute_metrics,
    latex_number,
    plot_confusion_matrix,
    plot_training_curves,
)
from model_builder import build_model

SPLIT_DIR = BASE_DIR / "dataset" / "split_series_v2"
OUTPUT_DIR = ARTIFACTS_DIR / "experiments" / "divisao_series_v2_cinza"
LUMINANCE = (0.299, 0.587, 0.114)  # pesos padrão (ITU-R BT.601) para converter RGB em tons de cinza


def build_gray_model(num_classes: int) -> tf.keras.Model:
    # Mesma arquitetura do model_builder.py, com uma camada fixa na entrada que converte a imagem para
    # tons de cinza (repetida nos 3 canais). A camada não é treinada e fica salva dentro do modelo, então
    # qualquer avaliação posterior também recebe a imagem em cinza.
    base = build_model(num_classes=num_classes)
    inputs = tf.keras.Input(shape=base.input_shape[1:], name="image")
    gray = layers.Conv2D(3, 1, use_bias=False, trainable=False, name="tons_de_cinza")
    x = gray(inputs)
    gray.set_weights([np.array(LUMINANCE, dtype="float32").reshape(1, 1, 3, 1).repeat(3, axis=3)])
    for layer in base.layers[1:]:
        x = layer(x)
    model = tf.keras.Model(inputs=inputs, outputs=x, name="coin_classifier_cinza")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina em tons de cinza com a divisão por série e avalia em séries nunca vistas.")
    parser.add_argument("--epocas", type=int, default=40, help="Número máximo de épocas (igual ao train.py)")
    parser.add_argument("--pasta", default=str(SPLIT_DIR),
                        help="Pasta com treino_validacao e teste (padrão: dataset/split_series_v2)")
    parser.add_argument("--saida", default=str(OUTPUT_DIR),
                        help="Pasta dos resultados (padrão: artifacts/experiments/divisao_series_v2_cinza)")
    parser.add_argument("--semente", type=int, default=SEED,
                        help="Semente do treino (padrão: a do config.py); a divisão treino/validação não muda")
    return parser.parse_args()


def evaluate(model, dataset, class_names) -> dict:
    y_true, y_pred = collect_predictions(model, dataset)
    matrix = build_confusion_matrix(y_true, y_pred, len(class_names))
    return compute_metrics(matrix, class_names)


def short(name: str) -> str:
    return name.split(",")[0]


def main() -> None:
    args = parse_args()
    split_dir, output_dir = Path(args.pasta), Path(args.saida)
    if not split_dir.exists():
        raise SystemExit(f"Pasta {split_dir} não encontrada. Rode antes: python make_group_split.py")
    if (output_dir / "resultados.json").exists():
        raise SystemExit(f"Já existe um resultado em {output_dir}. Apague essa pasta ou use outra --saida.")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Treino e validação: mesma divisão aleatória por imagem do projeto, só que dentro de treino_validacao
    train_ds, val_ds, class_names = build_datasets(dataset_dir=split_dir / "treino_validacao")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        split_dir / "teste",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )
    if list(test_ds.class_names) != list(class_names):
        raise ValueError("As classes do teste diferem das classes do treino.")

    tf.keras.utils.set_random_seed(args.semente)
    model = build_gray_model(num_classes=len(class_names))
    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epocas,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
        )],
        verbose=1,
    ).history
    minutes = (time.time() - start) / 60

    val_metrics = evaluate(model, val_ds, class_names)
    test_metrics = evaluate(model, test_ds, class_names)
    best_epoch = plot_training_curves(history, output_dir / "curvas_treinamento.png")
    plot_confusion_matrix(np.array(test_metrics["matriz_confusao"]), class_names, output_dir / "matriz_confusao_teste.png")
    model.save(output_dir / "modelo_divisao_series.keras")
    save_json(history, output_dir / "historico.json")
    save_json({
        "validacao_series_vistas": val_metrics,
        "teste_series_novas": test_metrics,
        "epocas_executadas": len(history["loss"]),
        "melhor_epoca": best_epoch,
        "minutos": round(minutes, 1),
        "semente": args.semente,
    }, output_dir / "resultados.json")

    rows = [("Validação (séries vistas)", val_metrics), ("Teste (séries novas)", test_metrics)]
    latex = [
        "\\begin{tabular}{@{}l" + "r" * (len(class_names) + 2) + "@{}}",
        "\\toprule",
        "Conjunto & Acurácia & F1 macro & " + " & ".join(f"Revocação {short(n)}" for n in class_names) + " \\\\",
        "\\midrule",
    ]
    for label, m in rows:
        recalls = " & ".join(latex_number(m["por_classe"][n]["revocacao"]) for n in class_names)
        acc = f"{m['acuracia'] * 100:.1f}".replace(".", ",") + "\\%"
        latex.append(f"{label} & {acc} & {latex_number(m['f1_macro'])} & {recalls} \\\\")
    latex += ["\\bottomrule", "\\end{tabular}"]
    (output_dir / "tabela_divisao_series.tex").write_text("\n".join(latex), encoding="utf-8")

    print("\nTONS DE CINZA - Divisão por série - validação (séries vistas no treino) x teste (séries nunca vistas)")
    print(f"Épocas executadas: {len(history['loss'])} (melhor época: {best_epoch}), {minutes:.0f} minutos\n")
    header = f"{'Conjunto':<28}{'Imagens':>9}{'Acurácia':>10}{'F1 macro':>10}" + "".join(f"{short(n):>20}" for n in class_names)
    print(header)
    for label, m in rows:
        recalls = "".join(f"{m['por_classe'][n]['revocacao'] * 100:>19.1f}%" for n in class_names)
        print(f"{label:<28}{m['total_imagens']:>9}{m['acuracia'] * 100:>9.1f}%{m['f1_macro']:>10.3f}{recalls}")
    print("(nas colunas das classes: revocação, ou seja, quanto de cada moeda foi reconhecido)")
    print(f"\nArquivos salvos em: {output_dir}")


if __name__ == "__main__":
    main()
