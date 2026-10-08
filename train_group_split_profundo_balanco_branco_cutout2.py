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
OUTPUT_DIR = ARTIFACTS_DIR / "experiments" / "divisao_series_v2_profundo_balanco_branco_cutout"
EXTRA_FILTERS = 256  # filtros de cada bloco novo
WB_PROBABILITY = 0.8                 # chance de uma foto do treino receber outra "cor de luz"
COLOR_TEMPERATURES = (2850.0, 7500.0)  # faixa padrão de iluminação: lâmpada incandescente (2850 K) até sombra (7500 K)
EXPOSURE = (0.6, 1.3)                # fator de brilho; no treino o brilho médio das séries varia mais de 2x (63 a 164)


def kelvin_to_rgb(kelvin: float) -> np.ndarray:
    # Cor aproximada de uma luz de corpo negro na temperatura dada (aproximação de Tanner Helland), em 0-255
    t = kelvin / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * np.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * np.log(t - 10) - 305.0447927307)
    return np.clip([r, g, b], 0.0, 255.0)


def white_balance_gains() -> np.ndarray:
    # Tabela de fatores (R, G, B) para 200 temperaturas, espaçadas igualmente em "mired" (1e6/K, a escala usada em
    # fotografia), relativas à luz do dia de referência (6500 K) e normalizadas para não mudar o brilho.
    temperatures = 1e6 / np.linspace(1e6 / COLOR_TEMPERATURES[1], 1e6 / COLOR_TEMPERATURES[0], 200)
    reference = kelvin_to_rgb(6500.0)
    gains = np.array([kelvin_to_rgb(t) / reference for t in temperatures])
    return gains / (gains @ np.array([0.299, 0.587, 0.114]))[:, None]


GAINS = tf.constant(white_balance_gains(), dtype=tf.float32)


def white_balance(images, labels):
    # Augmentation de balanço de branco (Afifi e Brown, ICCV 2019), só no TREINO: com 80% de chance, a foto é
    # "refotografada" com outra cor de luz, sorteada entre 2850 K (quente) e 7500 K (fria), e com outra exposição
    # (brilho x 0,6 a 1,3). Vale para as TRÊS moedas e o rótulo não muda: a cor da LUZ deixa de indicar a moeda.
    batch = tf.shape(images)[0]
    chosen = tf.cast(tf.random.uniform([batch]) < WB_PROBABILITY, images.dtype)[:, None]
    gains = tf.gather(GAINS, tf.random.uniform([batch], 0, GAINS.shape[0], dtype=tf.int32))
    exposure = tf.random.uniform([batch, 1], *EXPOSURE)
    factors = 1.0 + chosen * (gains * exposure - 1.0)
    return tf.clip_by_value(images * factors[:, None, None, :], 0.0, 255.0), labels


CUTOUT_SIZE = 56   # lado do quadrado apagado, em pixels (1/4 do lado da imagem de 224)
GRAY = 128.0       # cor do quadrado (cinza médio, na escala 0-255 das imagens carregadas)


def cutout(images, labels):
    # Cutout (DeVries e Taylor, 2017): em cada foto do TREINO, um quadrado cinza de 56x56 em posição aleatória
    # (pode ficar parcialmente fora da imagem). Validação e teste não passam por aqui. Como é feito nos dados,
    # e não numa camada do modelo, o modelo salvo é igual ao profundo e abre normalmente nos outros scripts.
    batch = tf.shape(images)[0]
    height, width = IMAGE_SIZE
    cy = tf.random.uniform([batch], 0, height, dtype=tf.int32)
    cx = tf.random.uniform([batch], 0, width, dtype=tf.int32)
    rows = tf.range(height)[None, :, None]
    cols = tf.range(width)[None, None, :]
    inside = (tf.abs(rows - cy[:, None, None]) < CUTOUT_SIZE // 2) & (tf.abs(cols - cx[:, None, None]) < CUTOUT_SIZE // 2)
    mask = tf.cast(inside, images.dtype)[..., None]
    return images * (1 - mask) + GRAY * mask, labels


def build_deep_model(num_classes: int) -> tf.keras.Model:
    # Mesma arquitetura do model_builder.py (mesmo aumento, mesmos 4 blocos, mesma parte final), com 2 blocos
    # a mais (MaxPooling + Conv2D 3x3) antes do GlobalAveragePooling. Com isso, cada unidade da última camada
    # passa a "ver" ~158x158 pixels da imagem de 224x224, em vez de 38x38.
    # (O modelo é IGUAL ao profundo; as diferenças estão só nos dados de treino: white_balance e cutout.)
    base = build_model(num_classes=num_classes)
    inputs = tf.keras.Input(shape=base.input_shape[1:], name="image")
    x = inputs
    for layer in base.layers[1:]:
        if isinstance(layer, layers.GlobalAveragePooling2D):
            for i in (5, 6):
                x = layers.MaxPooling2D(name=f"pool_bloco{i}")(x)
                x = layers.Conv2D(EXTRA_FILTERS, 3, padding="same", activation="relu", name=f"conv_bloco{i}")(x)
        x = layer(x)
    model = tf.keras.Model(inputs=inputs, outputs=x, name="coin_classifier_profundo_balanco_branco_cutout")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o modelo com 6 blocos + balanço de branco + cutout no treino na divisão por série e avalia em séries nunca vistas.")
    parser.add_argument("--epocas", type=int, default=40, help="Número máximo de épocas (igual ao train.py)")
    parser.add_argument("--pasta", default=str(SPLIT_DIR),
                        help="Pasta com treino_validacao e teste (padrão: dataset/split_series_v2)")
    parser.add_argument("--saida", default=str(OUTPUT_DIR),
                        help="Pasta dos resultados (padrão: artifacts/experiments/divisao_series_v2_profundo_balanco_branco_cutout)")
    parser.add_argument("--sem_parada", action="store_true",
                        help="Desliga o EarlyStopping: treina todas as --epocas")
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
    train_ds = train_ds.map(white_balance, num_parallel_calls=tf.data.AUTOTUNE)  # diferença 1: cor da luz
    train_ds = train_ds.map(cutout, num_parallel_calls=tf.data.AUTOTUNE)  # diferença 2: cutout (depois da cor, para o quadrado ficar cinza neutro)
    model = build_deep_model(num_classes=len(class_names))
    print(f"Parâmetros: {model.count_params():,}".replace(",", "."))
    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epocas,
        # Com --sem_parada, treina sempre todas as épocas e fica com os pesos da ÚLTIMA (sem EarlyStopping)
        callbacks=[] if args.sem_parada else [tf.keras.callbacks.EarlyStopping(
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
        "sem_parada": args.sem_parada,
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

    print("\nPROFUNDO + BALANÇO DE BRANCO + CUTOUT - Divisão por série - validação (séries vistas no treino) x teste (séries nunca vistas)")
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
