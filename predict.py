import argparse

import tensorflow as tf

from config import (
    CLASS_CENTERS_PATH,
    CLASS_NAMES_PATH,
    MODEL_PATH,
    OPENSET_CALIBRATION_PATH,
    TOP_K_TO_SHOW,
)
from dataset_loader import build_single_image_tensor
from io_utils import load_json
from model_builder import build_feature_extractor
from openset_utils import predict_with_unknown, sorted_percentages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prevê a moeda da imagem e também estima a chance de ser outra moeda fora do dataset."
    )
    parser.add_argument("--image", required=True, help="Caminho da imagem da moeda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = tf.keras.models.load_model(MODEL_PATH)
    feature_extractor = build_feature_extractor(model)

    class_names = load_json(CLASS_NAMES_PATH)
    class_centers = load_json(CLASS_CENTERS_PATH)
    calibration = load_json(OPENSET_CALIBRATION_PATH)

    image_tensor = build_single_image_tensor(args.image)
    results = predict_with_unknown(
        model=model,
        feature_extractor=feature_extractor,
        image_tensor=image_tensor,
        class_names=class_names,
        class_centers=class_centers,
        calibration=calibration,
    )

    ordered = sorted_percentages(results)
    predicted_class, predicted_prob = ordered[0]

    print("\nResultado da análise:")
    print(f"Classe mais provável: {predicted_class}")
    print(f"Confiança: {predicted_prob * 100:.2f}%")
    print("\nDistribuição das probabilidades:")

    for class_name, prob in ordered[:TOP_K_TO_SHOW]:
        print(f"- {class_name}: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
