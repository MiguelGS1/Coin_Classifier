import argparse
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from config import BASE_DIR
from io_utils import save_json

# Recorta as moedas do dataset LPR (Roboflow, Pascal VOC) e separa nas 3 classes do projeto.
# - Usa UMA cópia de cada foto original (o Roboflow gera cópias aumentadas com o mesmo nome base).
# - Fotos copiadas do U.S. Coins: a classe vem do nome do arquivo; as das SÉRIES DO TESTE v2 vão para a
#   subpasta "series_do_teste" (o montar_v2_extra.py não copia essa subpasta para o treino).
# - Outras fotos: a classe vem do nome no .xml; anos fora do período de cada classe são ignorados.
# Só lê a pasta baixada e salva numa pasta nova; não altera nada do projeto.
J, L, W = "Jefferson Nickels, 1938-Date", "Lincoln Cents, 1909-Date", "Washington Quarters, 1932-1998"
YEARS = {J: (1938, 2100), L: (1909, 2100), W: (1932, 1998)}
US_COINS = re.compile(r"^(Lincoln-Cent|Jefferson-Nickel|Washington-Quarter)-(\d+)_", re.IGNORECASE)
US_CLASS = {"lincoln-cent": L, "jefferson-nickel": J, "washington-quarter": W}
TEST_SERIES = {"30916", "26505", "212555", "107258", "41463", "72403", "17345"}
# Moedas de outros desenhos que NÃO são das nossas classes
OTHER_DESIGNS = re.compile(r"dime|buffalo|liberty|indian|shield|barber|seated|standing|state|park|"
                           r"territor|kennedy|half|eisenhower|sacagawea|susan|morgan|peace|flying|"
                           r"wartime|steel|large", re.IGNORECASE)
SPLIT_PREFERENCE = {"valid": 0, "test": 1, "train": 2}
MARGIN = 0.08
MIN_SIZE = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recorta as moedas do LPR por classe.")
    parser.add_argument("--entrada", default=str(BASE_DIR / "dataset" / "lpr_original"))
    parser.add_argument("--saida", default=str(BASE_DIR / "dataset" / "lpr_recortes"))
    return parser.parse_args()


def original_name(path: Path) -> str:
    return re.split(r"_(?:jpe?g|png)\.rf\.|\.rf\.", path.name)[0]


def class_from_label(label: str):
    text = label.lower()
    if OTHER_DESIGNS.search(text):
        return None
    if any(w in text for w in ("jefferson", "nickel", "5 cent", "5-cent", "five cent")):
        return J
    if any(w in text for w in ("washington", "quarter", "1/4", "1-4", "25 cent", "25-cent")):
        return W
    if any(w in text for w in ("lincoln", "penny", "1 cent", "1-cent", "one cent", "wheat", "memorial")):
        return L
    return None


def year_ok(cls: str, *texts: str) -> bool:
    years = [int(y) for t in texts for y in re.findall(r"(?<!\d)(1[789]\d\d|20\d\d)(?!\d)", t)]
    low, high = YEARS[cls]
    return all(low <= y <= high for y in years)


def main() -> None:
    args = parse_args()
    source, target = Path(args.entrada), Path(args.saida)
    if not source.exists():
        raise SystemExit(f"Pasta não encontrada: {source}")
    if target.exists():
        raise SystemExit(f"A pasta {target} já existe. Apague-a para recortar de novo.")

    chosen = {}
    for xml in sorted(source.rglob("*.xml")):
        key, rank = original_name(xml), SPLIT_PREFERENCE.get(xml.parent.name.lower(), 3)
        if key not in chosen or rank < chosen[key][0]:
            chosen[key] = (rank, xml)
    print(f"Fotos originais (sem cópias aumentadas): {len(chosen)}")

    kept, test_kept, ignored = Counter(), Counter(), Counter()
    labels_used = defaultdict(Counter)
    for key, (_, xml) in sorted(chosen.items()):
        image_path = next((p for p in xml.parent.glob(xml.stem + ".*") if p.suffix.lower() != ".xml"), None)
        if image_path is None:
            continue
        us = US_COINS.match(key)
        image = None
        for i, obj in enumerate(ET.parse(xml).getroot().iter("object")):
            label = obj.findtext("name", "").strip()
            # Caixas só da data ("date") ou da marca da casa da moeda ("D", "S", "P", "W") não são a moeda inteira
            if label.lower() == "date" or len(label) <= 2:
                ignored[label] += 1
                continue
            cls =US_CLASS[us.group(1).lower()] if us else class_from_label(label)
            box = obj.find("bndbox")
            x0, y0, x1, y1 = (float(box.findtext(t)) for t in ("xmin", "ymin", "xmax", "ymax"))
            if cls is None or not year_ok(cls, label, key) or min(x1 - x0, y1 - y0) < MIN_SIZE:
                ignored[label] += 1
                continue
            if us and us.group(1).lower() != "washington-quarter" and OTHER_DESIGNS.search(label):
                ignored[label] += 1
                continue
            image = image or Image.open(image_path).convert("RGB")
            mx, my = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
            crop = image.crop((max(0, x0 - mx), max(0, y0 - my),
                               min(image.width, x1 + mx), min(image.height, y1 + my)))
            side = max(crop.size)  # volta a moeda para redonda (o Roboflow esticou as fotos)
            in_test = bool(us) and us.group(2) in TEST_SERIES
            folder = target / ("series_do_teste" if in_test else "") / cls
            folder.mkdir(parents=True, exist_ok=True)
            crop.resize((side, side), Image.BICUBIC).save(folder / f"lpr_{key}_{i:02d}.jpg", quality=95)
            (test_kept if in_test else kept)[cls] += 1
            labels_used[cls][label] += 1

    save_json({"recortes_por_classe": dict(kept), "series_do_teste": dict(test_kept),
               "nomes_usados": {c: dict(v) for c, v in labels_used.items()}, "ignorados": dict(ignored)},
              target.parent / f"{target.name}_info.json")
    print("\nRecortes por moeda (vão para o treino):")
    for cls in (J, L, W):
        print(f"  {cls:<32} {kept[cls]:>5}")
    print(f"\nRecortes das séries do TESTE v2 (separados em 'series_do_teste', fora do treino): "
          f"{sum(test_kept.values())}")
    for cls in (J, L, W):
        print(f"\nNomes do .xml que viraram '{cls.split(',')[0]}':")
        for label, n in labels_used[cls].most_common(20):
            print(f"  {label:<45} {n:>5}")
    print(f"\nIgnorados (outras moedas/anos): {sum(ignored.values())} caixas, "
          f"{len(ignored)} nomes diferentes (lista completa no _info.json)")
    print(f"Recortes salvos em: {target}")


if __name__ == "__main__":
    main()
