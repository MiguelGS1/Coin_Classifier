import tensorflow as tf

from config import CLASS_CENTERS_PATH, CLASS_NAMES_PATH, MODEL_PATH, OPENSET_CALIBRATION_PATH
from dataset_loader import build_datasets
from io_utils import load_json
from model_builder import build_feature_extractor
from openset_utils import predict_with_unknown


def main() -> None:
    _, val_ds, class_names = build_datasets()
    model = tf.keras.models.load_model(MODEL_PATH)
    feature_extractor = build_feature_extractor(model)

    saved_class_names = load_json(CLASS_NAMES_PATH)
    if list(saved_class_names) != list(class_names):
        print("Aviso: as classes do dataset atual diferem das classes salvas no artefato.")

    class_centers = load_json(CLASS_CENTERS_PATH)
    calibration = load_json(OPENSET_CALIBRATION_PATH)

    correct_top1 = 0
    total = 0

    for batch_images, batch_labels in val_ds:
        true_indices = tf.argmax(batch_labels, axis=1).numpy()
        for i in range(batch_images.shape[0]):
            image_tensor = tf.expand_dims(batch_images[i], axis=0)
            results = predict_with_unknown(
                model=model,
                feature_extractor=feature_extractor,
                image_tensor=image_tensor,
                class_names=class_names,
                class_centers=class_centers,
                calibration=calibration,
            )
            predicted_class = max(results, key=results.get)
            true_class = class_names[int(true_indices[i])]
            if predicted_class == true_class:
                correct_top1 += 1
            total += 1

    accuracy = correct_top1 / total if total else 0.0
    print(f"Acurácia top-1 na validação: {accuracy * 100:.2f}%")
    print(
        "Observação: esta avaliação mede apenas o acerto nas classes conhecidas. "
        "A categoria 'outra_moeda' é estimada por lógica open-set."
    )


if __name__ == "__main__":
    main()
