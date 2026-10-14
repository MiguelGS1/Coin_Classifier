import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from config import ARTIFACTS_DIR, BASE_DIR, DATASET_DIR, IMAGE_SIZE

# Igual ao prever_imagem.py, mas antes de classificar encontra o círculo da moeda (OpenCV), recorta um quadrado em
# volta dela e pinta de branco tudo o que está FORA do círculo. Os pixels da moeda não são alterados.
# A imagem que o modelo recebe é salva para conferência. Não treina nada e não altera nenhum modelo.

EXPERIMENTS = ARTIFACTS_DIR / "experiments"
DEFAULT_MODELS = [
    EXPERIMENTS / "divisao_series_v2_profundo_balanco_branco_cutout" / "modelo_divisao_series.keras",
    EXPERIMENTS / "divisao_series_v2_profundo_balanco_branco_cutout_40ep_semente1" / "modelo_divisao_series.keras",
    EXPERIMENTS / "divisao_series_v2_profundo_balanco_branco_cutout_40ep_semente2" / "modelo_divisao_series.keras",
]
DEFAULT_NAMES = ["s42", "s1", "s2"]
DETECTION_SIDE = 400   # a detecção é feita numa cópia reduzida (maior lado = 400 px), como no crop_coins.py
MARGIN = 1.08          # o quadrado recortado tem lado 2 x 1,08 x raio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recorta a moeda, pinta o fundo de branco e classifica.")
    parser.add_argument("--imagem", required=True, nargs="+", help="Uma ou mais fotos (JPG, PNG, WEBP; HEIC se tiver pillow-heif)")
    parser.add_argument("--modelos", nargs="+", default=[str(p) for p in DEFAULT_MODELS],
                        help="Arquivos .keras (padrão: as 3 sementes do modelo final)")
    parser.add_argument("--nomes", nargs="+", default=None, help="Um nome curto por modelo")
    parser.add_argument("--fundo", type=int, default=255, help="Cor do fundo, 0-255 (padrão: 255 = branco; 128 = cinza)")
    parser.add_argument("--saida", default=str(BASE_DIR / "TESTES" / "recortadas"),
                        help="Pasta onde fica a imagem que o modelo recebeu (para conferir)")
    return parser.parse_args()


def load_rgb(path: str) -> np.ndarray:
    try:
        import pillow_heif  # opcional: permite abrir fotos .HEIC do iPhone
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    image = ImageOps.exif_transpose(Image.open(path))  # aplica a rotação gravada pelo celular
    return np.array(image.convert("RGB"))


def find_coin(rgb: np.ndarray):
    # Procura o círculo mais forte com raio entre 20% e 55% do menor lado (mesmos parâmetros do crop_coins.py)
    scale = DETECTION_SIDE / max(rgb.shape[:2])
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), 5)
    side = min(gray.shape)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=side, param1=100, param2=30,
                               minRadius=int(0.20 * side), maxRadius=int(0.55 * side))
    if circles is None:
        return None
    x, y, r = circles[0][0]
    return x / scale, y / scale, r / scale


def isolate_coin(rgb: np.ndarray, background: int):
    found = find_coin(rgb)
    height, width = rgb.shape[:2]
    if found is None:
        # Sem círculo: usa o quadrado central da foto, sem pintar o fundo
        side = min(height, width)
        top, left = (height - side) // 2, (width - side) // 2
        return rgb[top:top + side, left:left + side], False
    cx, cy, r = found
    half = int(round(MARGIN * r))
    # O quadrado pode passar da borda da foto; a parte de fora vira fundo
    canvas = np.full((2 * half, 2 * half, 3), background, dtype=np.uint8)
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + 2 * half, width), min(y0 + 2 * half, height)
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
    yy, xx = np.mgrid[0:2 * half, 0:2 * half]
    outside = (yy - (cy - y0)) ** 2 + (xx - (cx - x0)) ** 2 > r ** 2
    canvas[outside] = background
    return canvas, True


def main() -> None:
    args = parse_args()
    names = args.nomes or (DEFAULT_NAMES if args.modelos == [str(p) for p in DEFAULT_MODELS]
                           else [Path(m).parent.name[-12:] for m in args.modelos])
    if len(names) != len(args.modelos):
        raise SystemExit("Passe um nome para cada modelo (--nomes).")
    missing = [m for m in args.modelos if not Path(m).exists()]
    if missing:
        raise SystemExit("Modelos não encontrados:\n  " + "\n  ".join(missing))
    out_dir = Path(args.saida)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())
    short = [c.split()[0] for c in class_names]
    models = [tf.keras.models.load_model(m) for m in args.modelos]

    for path in args.imagem:
        coin, found = isolate_coin(load_rgb(path), args.fundo)
        seen = out_dir / f"{Path(path).stem}_recorte.png"
        Image.fromarray(coin).save(seen)
        # Mesmo redimensionamento do treino (bilinear para 224x224), valores 0-255; o modelo faz o Rescaling
        tensor = tf.image.resize(tf.constant(coin[None], dtype=tf.float32), IMAGE_SIZE)

        print(f"\nFoto: {path}")
        print(f"  {'Moeda encontrada e fundo pintado' if found else 'ATENÇÃO: círculo não encontrado; usado o quadrado central sem pintar o fundo'}")
        print(f"  Imagem que o modelo recebeu: {seen}")
        print(f"  {'':<12}" + "".join(f"{s:>12}" for s in short) + "   resposta")
        all_probs = []
        for name, model in zip(names, models):
            probs = model(tensor, training=False).numpy()[0]
            all_probs.append(probs)
            print(f"  {name:<12}" + "".join(f"{p * 100:>11.1f}%" for p in probs) + f"   {short[int(np.argmax(probs))]}")
        if len(models) > 1:
            mean = np.mean(all_probs, axis=0)
            print(f"  {'ENSEMBLE':<12}" + "".join(f"{p * 100:>11.1f}%" for p in mean) + f"   {class_names[int(np.argmax(mean))]}")


if __name__ == "__main__":
    main()
