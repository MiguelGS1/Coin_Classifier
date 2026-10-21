from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from config import (
    DISTANCE_KNOWN_PERCENTILE,
    MAX_UNKNOWN_PROB,
    MIN_UNKNOWN_PROB,
    SOFTMAX_KNOWN_PERCENTILE,
    UNKNOWN_CLASS_NAME,
    UNKNOWN_WEIGHT_DISTANCE,
    UNKNOWN_WEIGHT_SOFTMAX,
)


def _collect_embeddings_and_labels(
    feature_extractor: tf.keras.Model,
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_embeddings = []
    all_true_labels = []
    all_probs = []

    for batch_images, batch_labels in dataset:
        probs = model.predict(batch_images, verbose=0)
        embeddings = feature_extractor.predict(batch_images, verbose=0)
        true_indices = np.argmax(batch_labels.numpy(), axis=1)

        all_probs.append(probs)
        all_embeddings.append(embeddings)
        all_true_labels.append(true_indices)

    return (
        np.concatenate(all_embeddings, axis=0),
        np.concatenate(all_true_labels, axis=0),
        np.concatenate(all_probs, axis=0),
    )



def compute_class_centers(
    feature_extractor: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: List[str],
) -> Dict[str, List[float]]:
    embeddings_by_class = {class_name: [] for class_name in class_names}

    for batch_images, batch_labels in dataset:
        embeddings = feature_extractor.predict(batch_images, verbose=0)
        label_indices = np.argmax(batch_labels.numpy(), axis=1)

        for emb, idx in zip(embeddings, label_indices):
            embeddings_by_class[class_names[idx]].append(emb)

    class_centers = {}
    for class_name, vectors in embeddings_by_class.items():
        if not vectors:
            raise ValueError(f"Sem embeddings para a classe {class_name}.")
        center = np.mean(np.array(vectors), axis=0)
        class_centers[class_name] = center.tolist()

    return class_centers



def calibrate_open_set(
    feature_extractor: tf.keras.Model,
    model: tf.keras.Model,
    val_dataset: tf.data.Dataset,
    class_names: List[str],
    class_centers: Dict[str, List[float]],
) -> Dict[str, float]:
    centers = {k: np.array(v, dtype=np.float32) for k, v in class_centers.items()}
    embeddings, true_indices, probs = _collect_embeddings_and_labels(
        feature_extractor, model, val_dataset
    )

    true_confidences = probs[np.arange(len(probs)), true_indices]
    true_distances = []

    for emb, idx in zip(embeddings, true_indices):
        center = centers[class_names[idx]]
        dist = np.linalg.norm(emb - center)
        true_distances.append(dist)

    true_distances = np.array(true_distances, dtype=np.float32)

    softmax_threshold = float(
        np.percentile(true_confidences, SOFTMAX_KNOWN_PERCENTILE)
    )
    distance_threshold = float(
        np.percentile(true_distances, DISTANCE_KNOWN_PERCENTILE)
    )

    calibration = {
        "softmax_threshold": softmax_threshold,
        "distance_threshold": distance_threshold,
        "unknown_weight_softmax": UNKNOWN_WEIGHT_SOFTMAX,
        "unknown_weight_distance": UNKNOWN_WEIGHT_DISTANCE,
        "min_unknown_prob": MIN_UNKNOWN_PROB,
        "max_unknown_prob": MAX_UNKNOWN_PROB,
        "unknown_class_name": UNKNOWN_CLASS_NAME,
    }
    return calibration



def _unknown_score_from_softmax(max_prob: float, threshold: float) -> float:
    if max_prob >= threshold:
        return 0.0
    if threshold <= 1e-8:
        return 0.0
    score = (threshold - max_prob) / threshold
    return float(np.clip(score, 0.0, 1.0))



def _unknown_score_from_distance(distance: float, threshold: float) -> float:
    if distance <= threshold:
        return 0.0
    denom = max(distance, threshold, 1e-8)
    score = (distance - threshold) / denom
    return float(np.clip(score, 0.0, 1.0))



def predict_with_unknown(
    model: tf.keras.Model,
    feature_extractor: tf.keras.Model,
    image_tensor: tf.Tensor,
    class_names: List[str],
    class_centers: Dict[str, List[float]],
    calibration: Dict[str, float],
) -> Dict[str, float]:
    probs = model.predict(image_tensor, verbose=0)[0]
    embedding = feature_extractor.predict(image_tensor, verbose=0)[0]

    centers = {k: np.array(v, dtype=np.float32) for k, v in class_centers.items()}
    distances = {
        name: float(np.linalg.norm(embedding - center)) for name, center in centers.items()
    }

    max_prob = float(np.max(probs))
    min_distance = min(distances.values())

    s_score = _unknown_score_from_softmax(max_prob, calibration["softmax_threshold"])
    d_score = _unknown_score_from_distance(
        min_distance, calibration["distance_threshold"]
    )

    unknown_prob = (
        calibration["unknown_weight_softmax"] * s_score
        + calibration["unknown_weight_distance"] * d_score
    )
    unknown_prob = float(
        np.clip(
            unknown_prob,
            calibration["min_unknown_prob"],
            calibration["max_unknown_prob"],
        )
    )

    known_scale = 1.0 - unknown_prob
    adjusted_probs = probs * known_scale

    results = {class_name: float(prob) for class_name, prob in zip(class_names, adjusted_probs)}
    results[calibration.get("unknown_class_name", UNKNOWN_CLASS_NAME)] = unknown_prob

    total = sum(results.values())
    if total > 0:
        results = {k: v / total for k, v in results.items()}

    return results



def sorted_percentages(results: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(results.items(), key=lambda item: item[1], reverse=True)
