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
OUTPUT_DIR = ARTIFACTS_DIR / "experiments" / "divisao_series_v2_profundo_dourado"
EXTRA_FILTERS = 256  # filtros de cada bloco novo
GILD_PROBABILITY = 0.5           # chance de uma foto do treino receber o tom quente
GILD_STRENGTH = (0.1, 0.5)       # força do tom quente, sorteada para cada foto
GILD_CHANNELS = (1.0, 0.4, -1.0)  # quanto cada canal muda por unidade de força: R sobe, G sobe menos, B desce


def gild(images, labels):
    # "Douramento" (só no TREINO): com 50% de chance, a foto recebe um tom quente (R x (1+a), G x (1+0,4a),
    # B x (1-a), com a entre 0,1 e 0,5), simulando pátina dourada e luz amarelada. Vale para as TRÊS moedas e o
    # rótulo não muda. Assim o "tom quente" fica independente da classe: uma prateada dourada continua prateada,
    # e um Lincoln com luz quente continua Lincoln. Se só as prateadas fossem douradas, "foto amarelada" viraria
    # uma pista falsa de prateada. Ideia de augmentation direcionada a um atalho (targeted augmentation;
    # Gao et al., ICML 2023): variar só o atributo que é atalho, sem apagar a informação real (o cobre do Lincoln).
    batch = tf.shape(images)[0]
    chosen = tf.random.uniform([batch]) < GILD_PROBABILITY
    strength = tf.random.uniform([batch], *GILD_STRENGTH) * tf.cast(chosen, images.dtype)
    factors = 1.0 + strength[:, None] * tf.constant(GILD_CHANNELS, dtype=images.dtype)[None, :]
    return tf.clip_by_value(images * factors[:, None, None, :], 0.0, 255.0), labels


def build_deep_model(num_classes: int) -> tf.keras.Model:
    # Mesma arquitetura do model_builder.py (mesmo aumento, mesmos 4 blocos, mesma parte final), com 2 blocos
    # a mais (MaxPooling + Conv2D 3x3) antes do GlobalAveragePooling. Com isso, cada unidade da última camada
    # passa a "ver" ~158x158 pixels da imagem de 224x224, em vez de 38x38.
    # (O modelo é IGUAL ao profundo; a diferença está só nos dados de treino, na função gild.)
    base = build_model(num_classes=num_classes)
    inputs = tf.keras.Input(shape=base.input_shape[1:], name="image")
    x = inputs
    for layer in base.layers[1:]:
        if isinstance(layer, layers.GlobalAveragePooling2D):
            for i in (5, 6):
                x = layers.MaxPooling2D(name=f"pool_bloco{i}")(x)
                x = layers.Conv2D(EXTRA_FILTERS, 3, padding="same", activation="relu", name=f"conv_bloco{i}")(x)
        x = layer(x)
    model = tf.keras.Model(inputs=inputs, outputs=x, name="coin_classifier_profundo_dourado")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o modelo com 6 blocos + douramento (tom quente aleatório) no treino na divisão por série e avalia em séries nunca vistas.")
    parser.add_argument("--epocas", type=int, default=40, help="Número máximo de épocas (igual ao train.py)")
    parser.add_argument("--pasta", default=str(SPLIT_DIR),
                        help="Pasta com treino_validacao e teste (padrão: dataset/split_series_v2)")
    parser.add_argument("--saida", default=str(OUTPUT_DIR),
                        help="Pasta dos resultados (padrão: artifacts/experiments/divisao_series_v2_profundo_dourado)")
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
    train_ds = train_ds.map(gild, num_parallel_calls=tf.data.AUTOTUNE)  # ÚNICA diferença para o profundo
    model = build_deep_model(num_classes=len(class_names))
    print(f"Parâmetros: {model.count_params():,}".replace(",", "."))
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

    print("\nPROFUNDO + DOURAMENTO (tom quente) - Divisão por série - validação (séries vistas no treino) x teste (séries nunca vistas)")
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
