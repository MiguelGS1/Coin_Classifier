import argparse
import time

import tensorflow as tf
from tensorflow.keras import layers

from config import CLASS_NAMES_PATH, EARLY_STOPPING_PATIENCE, LEARNING_RATE, SEED
from dataset_loader import build_datasets
from io_utils import load_json, save_json
from model_builder import build_model
from testar_celular import OUTPUT_DIR, testar

MODELS_DIR = OUTPUT_DIR / "modelos"


def atual():
    # As mesmas camadas de aumento do model_builder.py
    return [layers.RandomFlip("horizontal"), layers.RandomRotation(0.05),
            layers.RandomZoom(0.08), layers.RandomContrast(0.08)]


def luz_cor():
    # Imagens já estão entre 0 e 1 (depois do Rescaling), por isso value_range=(0, 1)
    return [layers.RandomBrightness(0.3, value_range=(0, 1)),
            layers.RandomContrast(0.3),
            layers.RandomSaturation(0.3, value_range=(0, 1)),
            layers.RandomHue(0.05, value_range=(0, 1)),
            layers.RandomGrayscale(0.2)]


# Lista fixa de variantes, definida antes de ver os resultados
VARIANTS = {
    "sem_aumento": lambda: [],
    "rotacao": lambda: [layers.RandomFlip("horizontal"), layers.RandomRotation(0.5),
                        layers.RandomZoom(0.08), layers.RandomContrast(0.08)],
    "luz_cor": lambda: [layers.RandomFlip("horizontal"), layers.RandomRotation(0.05),
                        layers.RandomZoom(0.08)] + luz_cor(),
    "forte": lambda: [layers.RandomFlip("horizontal"), layers.RandomRotation(0.5),
                      layers.RandomZoom(0.2)] + luz_cor()
                     + [layers.RandomGaussianBlur(0.5, kernel_size=5, sigma=(0.1, 1.5), value_range=(0, 1))],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o modelo com outro aumento de dados e testa nas fotos de celular.")
    parser.add_argument("--variante", required=True, choices=list(VARIANTS))
    parser.add_argument("--epocas", type=int, default=40, help="Número máximo de épocas (igual ao train.py)")
    return parser.parse_args()


def build_variant_model(num_classes: int, augmentation: list) -> tf.keras.Model:
    # Mesma arquitetura do model_builder.py; só troca as camadas aug_* pelas da variante
    base = build_model(num_classes=num_classes)
    inputs = tf.keras.Input(shape=base.input_shape[1:], name="image")
    x = inputs
    for layer in base.layers[1:]:
        if layer.name.startswith("aug_"):
            continue
        x = layer(x)
        if layer.name == "rescale":
            for aug in augmentation:
                x = aug(x)
    model = tf.keras.Model(inputs=inputs, outputs=x, name="coin_classifier")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    args = parse_args()
    model_path = MODELS_DIR / f"{args.variante}.keras"
    if model_path.exists():
        raise SystemExit(f"A variante '{args.variante}' já foi treinada ({model_path}). Apague o arquivo para treinar de novo.")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, class_names = build_datasets()
    if list(class_names) != list(load_json(CLASS_NAMES_PATH)):
        raise ValueError("As classes do dataset diferem das do modelo final.")

    tf.keras.utils.set_random_seed(SEED)
    model = build_variant_model(len(class_names), VARIANTS[args.variante]())
    print(f"\nVariante '{args.variante}', camadas de aumento:",
          [l.__class__.__name__ for l in model.layers[2:] if l.__class__.__name__.startswith("Random")] or "nenhuma")

    start = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epocas, verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                                                    restore_best_weights=True)],
    ).history
    model.save(model_path)
    save_json(history, MODELS_DIR / f"{args.variante}_historico.json")
    best = int(min(range(len(history["val_loss"])), key=history["val_loss"].__getitem__)) + 1
    print(f"\nTreino concluído em {(time.time() - start) / 60:.0f} minutos "
          f"({len(history['loss'])} épocas, melhor época: {best}). Testando...")

    testar(model, args.variante, list(class_names))


if __name__ == "__main__":
    main()
