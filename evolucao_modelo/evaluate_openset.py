import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import (
    ARTIFACTS_DIR,
    BASE_DIR,
    CLASS_CENTERS_PATH,
    CLASS_NAMES_PATH,
    MODEL_PATH,
    OPENSET_CALIBRATION_PATH,
)
from dataset_loader import build_datasets, build_single_image_tensor
from io_utils import load_json, save_json
from model_builder import build_feature_extractor
from openset_utils import _unknown_score_from_distance, _unknown_score_from_softmax

DEFAULT_UNKNOWN_DIR = BASE_DIR / "dataset" / "unknown"
DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "experiments" / "exp_c"
UNKNOWN_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CHUNK_SIZE = 32

# Taxa de alarmes falsos usada para comparar os sinais com o mesmo limiar
FALSE_ALARM_TARGET = 0.10

SIGNAL_NAMES = {
    "confianca": "Confiança",
    "distancia": "Distância",
    "combinado": "Combinado",
}


def compute_signals(model, feature_extractor, images, centers, calibration) -> dict:
    # Mesmos cálculos de predict_with_unknown, feitos para um lote de imagens
    probs = model.predict(images, verbose=0)
    embeddings = feature_extractor.predict(images, verbose=0)

    max_prob = probs.max(axis=1)
    distances = np.linalg.norm(embeddings[:, None, :] - centers[None, :, :], axis=2)
    min_distance = distances.min(axis=1)

    unknown_prob = []
    for c, d in zip(max_prob, min_distance):
        s_c = _unknown_score_from_softmax(float(c), calibration["softmax_threshold"])
        s_d = _unknown_score_from_distance(float(d), calibration["distance_threshold"])
        u = (
            calibration["unknown_weight_softmax"] * s_c
            + calibration["unknown_weight_distance"] * s_d
        )
        unknown_prob.append(
            np.clip(u, calibration["min_unknown_prob"], calibration["max_unknown_prob"])
        )
    unknown_prob = np.array(unknown_prob)

    # O sistema responde "outra_moeda" quando ela fica em 1º lugar
    system_rejects = unknown_prob > max_prob * (1 - unknown_prob)

    return {
        "confianca": 1 - max_prob,  # quanto maior, mais parece desconhecida
        "distancia": min_distance,
        "combinado": unknown_prob,
        "sistema_rejeita": system_rejects,
    }


def merge_signals(parts: list[dict]) -> dict:
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def signals_for_known(model, feature_extractor, val_ds, centers, calibration) -> dict:
    parts = [
        compute_signals(model, feature_extractor, batch_images, centers, calibration)
        for batch_images, _ in val_ds
    ]
    return merge_signals(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimento C: rejeição de moedas desconhecidas.")
    parser.add_argument("--unknown-dir", default=str(DEFAULT_UNKNOWN_DIR),
                        help="Pasta com as imagens desconhecidas (uma subpasta por tipo)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR),
                        help="Pasta onde os resultados serão salvos")
    return parser.parse_args()


def find_unknown_images(unknown_dir: Path) -> list:
    if not unknown_dir.exists():
        raise FileNotFoundError(
            f"Pasta de imagens desconhecidas não encontrada: {unknown_dir}\n"
            "Crie a pasta com uma subpasta para cada tipo de imagem."
        )
    paths = sorted(
        p for p in unknown_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in UNKNOWN_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"Nenhuma imagem .jpg, .jpeg ou .png em {unknown_dir}")
    return paths


def unknown_type(path, unknown_dir: Path) -> str:
    relative = path.relative_to(unknown_dir)
    return relative.parts[0] if len(relative.parts) > 1 else "(sem subpasta)"


def signals_for_unknown(model, feature_extractor, paths, centers, calibration) -> dict:
    parts = []
    for start in range(0, len(paths), CHUNK_SIZE):
        chunk = paths[start:start + CHUNK_SIZE]
        images = tf.concat([build_single_image_tensor(str(p)) for p in chunk], axis=0)
        parts.append(compute_signals(model, feature_extractor, images, centers, calibration))
    return merge_signals(parts)


def auroc(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    # Chance de uma desconhecida ter score maior que uma conhecida (empate vale meio)
    greater = (unknown_scores[:, None] > known_scores[None, :]).mean()
    ties = (unknown_scores[:, None] == known_scores[None, :]).mean()
    return float(greater + 0.5 * ties)


def roc_curve(known_scores: np.ndarray, unknown_scores: np.ndarray):
    thresholds = np.unique(np.concatenate([known_scores, unknown_scores]))[::-1]
    false_alarm = [0.0] + [float((known_scores >= t).mean()) for t in thresholds]
    rejection = [0.0] + [float((unknown_scores >= t).mean()) for t in thresholds]
    return false_alarm, rejection


def evaluate_signal(known_scores: np.ndarray, unknown_scores: np.ndarray) -> dict:
    # Limiar que deixa cerca de 10% das moedas conhecidas acima dele
    threshold = float(np.percentile(known_scores, 100 * (1 - FALSE_ALARM_TARGET)))
    return {
        "auroc": auroc(known_scores, unknown_scores),
        "limiar": threshold,
        "rejeicao_desconhecidas": float((unknown_scores > threshold).mean()),
        "alarmes_falsos": float((known_scores > threshold).mean()),
    }


def plot_roc(known: dict, unknown: dict, results: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for key, label in SIGNAL_NAMES.items():
        false_alarm, rejection = roc_curve(known[key], unknown[key])
        auc_text = f"{results[key]['auroc']:.2f}".replace(".", ",")
        ax.plot(false_alarm, rejection, label=f"{label} (AUROC = {auc_text})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Taxa de alarmes falsos (moedas conhecidas)")
    ax.set_ylabel("Taxa de rejeição (moedas desconhecidas)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def latex_number(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def latex_percent(value: float) -> str:
    return f"{value * 100:.1f}".replace(".", ",") + "\\%"


def build_latex_table(results: dict, system: dict) -> str:
    lines = [
        "\\begin{tabular}{@{}lrrr@{}}",
        "\\toprule",
        "Sinal & AUROC & Rejeição & Alarmes falsos \\\\",
        "\\midrule",
    ]
    for key, label in SIGNAL_NAMES.items():
        values = results[key]
        lines.append(
            f"{label} & {latex_number(values['auroc'])} & "
            f"{latex_percent(values['rejeicao_desconhecidas'])} & "
            f"{latex_percent(values['alarmes_falsos'])} \\\\"
        )
    lines += [
        "\\midrule",
        "Sistema (regra atual) & -- & "
        f"{latex_percent(system['rejeicao_desconhecidas'])} & "
        f"{latex_percent(system['alarmes_falsos'])} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    unknown_dir = Path(args.unknown_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, val_ds, class_names = build_datasets()
    saved_class_names = load_json(CLASS_NAMES_PATH)
    if list(saved_class_names) != list(class_names):
        raise ValueError(
            "As classes do dataset atual diferem das classes do modelo treinado. "
            "Rode python train.py novamente antes do experimento."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    feature_extractor = build_feature_extractor(model)
    class_centers = load_json(CLASS_CENTERS_PATH)
    centers = np.array([class_centers[name] for name in class_names], dtype=np.float32)
    calibration = load_json(OPENSET_CALIBRATION_PATH)

    unknown_paths = find_unknown_images(unknown_dir)
    known = signals_for_known(model, feature_extractor, val_ds, centers, calibration)
    unknown = signals_for_unknown(model, feature_extractor, unknown_paths, centers, calibration)

    results = {key: evaluate_signal(known[key], unknown[key]) for key in SIGNAL_NAMES}
    system = {
        "rejeicao_desconhecidas": float(unknown["sistema_rejeita"].mean()),
        "alarmes_falsos": float(known["sistema_rejeita"].mean()),
    }

    types = np.array([unknown_type(p, unknown_dir) for p in unknown_paths])
    combined_threshold = results["combinado"]["limiar"]
    per_type = {}
    for name in sorted(set(types)):
        mask = types == name
        per_type[name] = {
            "imagens": int(mask.sum()),
            "rejeicao_combinado": float((unknown["combinado"][mask] > combined_threshold).mean()),
            "rejeicao_sistema": float(unknown["sistema_rejeita"][mask].mean()),
        }

    report = {
        "imagens_conhecidas": int(len(known["combinado"])),
        "imagens_desconhecidas": len(unknown_paths),
        "pasta_desconhecidas": str(unknown_dir),
        "meta_alarmes_falsos": FALSE_ALARM_TARGET,
        "sinais": results,
        "sistema": system,
        "por_tipo_desconhecido": per_type,
        "calibracao": calibration,
    }

    plot_roc(known, unknown, results, output_dir / "curva_roc.png")
    save_json(report, output_dir / "metricas_openset.json")
    (output_dir / "tabela_openset.tex").write_text(
        build_latex_table(results, system), encoding="utf-8"
    )

    print("\nExperimento C - reconhecimento open-set")
    print(f"Moedas conhecidas (validação): {report['imagens_conhecidas']}")
    print(f"Moedas desconhecidas: {report['imagens_desconhecidas']}\n")
    print(f"{'Sinal':<12}{'AUROC':>8}{'Rejeição':>11}{'Alarmes falsos':>16}")
    for key, label in SIGNAL_NAMES.items():
        values = results[key]
        print(
            f"{label:<12}{values['auroc']:>8.3f}"
            f"{values['rejeicao_desconhecidas'] * 100:>10.1f}%"
            f"{values['alarmes_falsos'] * 100:>15.1f}%"
        )
    print(
        f"{'Sistema':<12}{'-':>8}{system['rejeicao_desconhecidas'] * 100:>10.1f}%"
        f"{system['alarmes_falsos'] * 100:>15.1f}%"
    )
    print("\nRejeição por tipo de imagem desconhecida:")
    print(f"{'Tipo':<35}{'Imagens':>9}{'Combinado':>11}{'Sistema':>10}")
    for name, values in per_type.items():
        print(
            f"{name[:34]:<35}{values['imagens']:>9}"
            f"{values['rejeicao_combinado'] * 100:>10.1f}%"
            f"{values['rejeicao_sistema'] * 100:>9.1f}%"
        )
    print(f"\nArquivos salvos em: {output_dir}")


if __name__ == "__main__":
    main()
