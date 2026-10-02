import csv
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from config import ARTIFACTS_DIR, BASE_DIR, BATCH_SIZE, CLASS_NAMES_PATH, IMAGE_SIZE
from degradacoes_validacao import MODELS
from io_utils import load_json, save_json

CELULAR_DIR = BASE_DIR / "dataset" / "celular" / "teste_celular"
OUTPUT_DIR = ARTIFACTS_DIR / "experiments" / "fatores_celular"
GROUPS = ["baixo", "medio", "alto"]


def measures(img):
    # img: RGB 224x224, 0-255
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return {
        "nitidez": float(cv2.Laplacian(gray, cv2.CV_64F).var()),  # menor = mais borrada
        "brilho": float(gray.mean()),                             # menor = mais escura
        "tom": float(img[..., 0].mean() - img[..., 2].mean()),    # maior = mais amarelada/avermelhada
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    class_names = list(load_json(CLASS_NAMES_PATH))
    ds = tf.keras.utils.image_dataset_from_directory(
        CELULAR_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="int",
        class_names=class_names, shuffle=False,
    )
    paths = list(ds.file_paths)
    images, labels = [], []
    for x, y in ds:
        images.append(np.clip(x.numpy(), 0, 255).astype(np.uint8))
        labels.append(y.numpy())
    images, labels = np.concatenate(images), np.concatenate(labels)
    feats = [measures(img) for img in images]

    preds = {}
    for name, path in MODELS.items():
        if not path.exists():
            print(f"Pulando '{name}': modelo não encontrado em {path}")
            continue
        print(f"Avaliando o modelo '{name}'...")
        model = tf.keras.models.load_model(path)
        out = [np.argmax(model(tf.constant(images[s:s + 32], dtype=tf.float32), training=False).numpy(), axis=1)
               for s in range(0, len(images), 32)]
        preds[name] = np.concatenate(out)

    # Grupos (terços) calculados DENTRO de cada classe: o Lincoln é naturalmente mais avermelhado,
    # então dividir todas juntas misturaria o efeito do fator com o da classe
    group_of = {f: np.empty(len(images), dtype=object) for f in ("nitidez", "brilho", "tom")}
    for f in group_of:
        values = np.array([d[f] for d in feats])
        for c in range(len(class_names)):
            idx = np.where(labels == c)[0]
            cuts = np.percentile(values[idx], [100 / 3, 200 / 3])
            group_of[f][idx] = np.array(GROUPS)[np.searchsorted(cuts, values[idx], side="right")]

    with open(OUTPUT_DIR / "por_imagem.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["arquivo", "classe", "nitidez", "brilho", "tom", "grupo_nitidez", "grupo_brilho", "grupo_tom"]
                   + [f"previsto_{m}" for m in preds])
        for i, p in enumerate(paths):
            w.writerow([Path(p).name, class_names[labels[i]], f"{feats[i]['nitidez']:.1f}",
                        f"{feats[i]['brilho']:.1f}", f"{feats[i]['tom']:.1f}",
                        group_of["nitidez"][i], group_of["brilho"][i], group_of["tom"][i]]
                       + [class_names[preds[m][i]].split()[0] for m in preds])

    lincoln = class_names.index("Lincoln Cents, 1909-Date")
    silver = labels != lincoln
    summary = {}
    print("\nAcurácia balanceada (%) por terço de cada medida (terços calculados dentro de cada classe)")
    print("e, entre parênteses, % das moedas prateadas classificadas como Lincoln")
    for f, desc in (("nitidez", "baixo = mais borrada"), ("brilho", "baixo = mais escura"),
                    ("tom", "alto = mais amarelada")):
        print(f"\n{f} ({desc})")
        print(f"  {'modelo':<12}" + "".join(f"{g:>18}" for g in GROUPS))
        summary[f] = {}
        for m, pr in preds.items():
            cells, summary[f][m] = [], {}
            for g in GROUPS:
                sel = group_of[f] == g
                recalls = [float((pr[sel & (labels == c)] == c).mean()) for c in range(len(class_names))]
                bal = float(np.mean(recalls))
                to_lincoln = float((pr[sel & silver] == lincoln).mean())
                summary[f][m][g] = {"balanceada": bal, "acerto_por_classe": dict(zip(class_names, recalls)),
                                    "prateadas_como_lincoln": to_lincoln, "imagens": int(sel.sum())}
                cells.append(f"{bal * 100:>9.1f} ({to_lincoln * 100:>4.0f}%)")
            print(f"  {m:<12}" + "".join(f"{c:>18}" for c in cells))
    counts = {g: {c.split()[0]: int(((group_of['nitidez'] == g) & (labels == i)).sum())
                  for i, c in enumerate(class_names)} for g in GROUPS}
    print(f"\nImagens por terço de nitidez (as outras medidas têm contagens iguais ou muito próximas): {counts}")
    save_json(summary, OUTPUT_DIR / "resumo.json")
    print(f"Arquivos salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
