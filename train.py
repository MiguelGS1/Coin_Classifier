from pathlib import Path

import tensorflow as tf

from config import (
    ARTIFACTS_DIR,
    CLASS_CENTERS_PATH,
    CLASS_NAMES_PATH,
    MODEL_PATH,
    OPENSET_CALIBRATION_PATH,
    TRAINING_HISTORY_PATH,
    EARLY_STOPPING_PATIENCE,
)
from dataset_loader import build_datasets
from io_utils import save_json
from model_builder import build_feature_extractor, build_model
from openset_utils import calibrate_open_set, compute_class_centers


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, class_names = build_datasets()
    num_classes = len(class_names)

    print("\nClasses encontradas:")
    for idx, class_name in enumerate(class_names):
        print(f"  {idx}: {class_name}")

    model = build_model(num_classes=num_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=18,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(MODEL_PATH)
    save_json(class_names, CLASS_NAMES_PATH)
    save_json(history.history, TRAINING_HISTORY_PATH)

    feature_extractor = build_feature_extractor(model)
    class_centers = compute_class_centers(feature_extractor, train_ds, class_names)
    save_json(class_centers, CLASS_CENTERS_PATH)

    calibration = calibrate_open_set(
        feature_extractor=feature_extractor,
        model=model,
        val_dataset=val_ds,
        class_names=class_names,
        class_centers=class_centers,
    )
    save_json(calibration, OPENSET_CALIBRATION_PATH)

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print("\nTreino concluído.")
    print(f"Val loss: {val_loss:.4f}")
    print(f"Val accuracy: {val_acc:.4f}")
    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Centros das classes salvos em: {CLASS_CENTERS_PATH}")
    print(f"Calibração open-set salva em: {OPENSET_CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
