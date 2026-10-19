import argparse
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from config import BASE_DIR
from io_utils import save_json

# Recorta as moedas de um dataset do Roboflow (formato Pascal VOC) e separa por classe.
# - Usa UMA cópia de cada foto original (o Roboflow gera várias cópias aumentadas com o mesmo nome base;
#   dá preferência à cópia da pasta "valid", que não tem aumento).
# - Ignora dimes e classes que não são penny/nickel/quarter.
# - Compara cada recorte com as fotos de teste (v2 e celular) e separa os muito parecidos.
# Só lê a pasta baixada e salva numa pasta nova; não altera nada do projeto.
# Palavras do nome da classe no .xml -> classe do projeto (ex.: "Penny", "Lincoln Cents Dollar obverse")
CLASS_OF = {("jefferson", "nickel"): "Jefferson Nickels, 1938-Date",
            ("lincoln", "penny", "cent"): "Lincoln Cents, 1909-Date",
            ("washington", "quarter"): "Washington Quarters, 1932-1998"}
SPLIT_PREFERENCE = {"valid": 0, "test": 1, "train": 2}
MARGIN = 0.08          # margem em volta da caixa (fração do lado)
MIN_SIZE = 40          # caixas menores que isso (em pixels) são ignoradas
HASH_LIMIT = 6         # distância máxima do "hash visual" para considerar duas imagens quase iguais


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recorta as moedas de um dataset Roboflow (Pascal VOC) por classe.")
    parser.add_argument("--entrada", required=True, help="Pasta descompactada do Roboflow (com train/valid/test)")
    parser.add_argument("--saida", required=True, help="Pasta nova para os recortes")
    return parser.parse_args()


def original_name(path: Path) -> str:
    # "076225ca-20220629_181244_jpg.rf.31fd...jpg" -> "20220629_181244"; "coins_39_jpeg.rf.a67c...jpg" -> "coins_39"
    stem = re.split(r"_(?:jpe?g|png)\.rf\.|\.rf\.", path.name)[0]
    return re.sub(r"^[0-9a-f]{8}-", "", stem)


def coin_class(name: str):
    name = name.strip().lower()
    if "dime" in name:
        return None
    for words, cls in CLASS_OF.items():
        if any(w in name for w in words):
            return cls
    return None


def dhash(image: Image.Image) -> np.ndarray:
    small = np.asarray(image.convert("L").resize((9, 8)), dtype=np.int16)
    return (small[:, 1:] > small[:, :-1]).flatten()


def test_hashes(folders: list) -> list:
    hashes = []
    for folder in folders:
        if folder.exists():
            for p in folder.rglob("*"):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    hashes.append((dhash(Image.open(p)), str(p)))
    return hashes


def main() -> None:
    args = parse_args()
    source, target = Path(args.entrada), Path(args.saida)
    if not source.exists():
        raise SystemExit(f"Pasta não encontrada: {source}")
    if target.exists():
        raise SystemExit(f"A pasta {target} já existe. Apague-a para recortar de novo.")

    # Escolhe uma cópia por foto original
    chosen = {}
    for xml in sorted(source.rglob("*.xml")):
        split = xml.parent.name.lower()
        key = original_name(xml)
        rank = SPLIT_PREFERENCE.get(split, 3)
        if key not in chosen or rank < chosen[key][0]:
            chosen[key] = (rank, xml)
    print(f"Arquivos .xml: {len(list(source.rglob('*.xml')))}   fotos originais (sem cópias): {len(chosen)}")

    names_seen, kept, ignored = Counter(), Counter(), Counter()
    crops = []
    for key, (_, xml) in sorted(chosen.items()):
        image_path = xml.with_suffix(".jpg")
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        for i, obj in enumerate(ET.parse(xml).getroot().iter("object")):
            name = obj.findtext("name", "")
            names_seen[name] += 1
            cls = coin_class(name)
            box = obj.find("bndbox")
            x0, y0, x1, y1 = (float(box.findtext(t)) for t in ("xmin", "ymin", "xmax", "ymax"))
            if cls is None or min(x1 - x0, y1 - y0) < MIN_SIZE:
                ignored[name] += 1
                continue
            # Recorte da caixa com margem; depois a imagem é redimensionada para quadrado. Como o Roboflow
            # esticou as fotos para 640x640 ("Stretch"), as moedas ficaram ovais e a caixa não é quadrada:
            # deixar o recorte quadrado devolve o formato redondo da moeda.
            mx, my = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
            left, top = max(0, x0 - mx), max(0, y0 - my)
            right, bottom = min(image.width, x1 + mx), min(image.height, y1 + my)
            crop = image.crop((left, top, right, bottom))
            side = max(crop.size)
            crops.append((cls, f"{source.name}_{key}_{i:02d}.jpg", crop.resize((side, side), Image.BICUBIC), image))
            kept[cls] += 1

    # Compara com as fotos de teste (divisão v2 e celular)
    references = test_hashes([BASE_DIR / "dataset" / "split_series_v2" / "teste",
                              BASE_DIR / "dataset" / "celular" / "teste_celular"])
    suspicious = []
    for cls, name, crop, full in crops:
        # compara o recorte E a foto inteira (em datasets de uma moeda por foto, a foto inteira é a que se parece
        # com as fotos do U.S. Coins)
        hashes = [dhash(crop), dhash(full)]
        close = [ref for ref_hash, ref in references
                 if min(int(np.sum(h != ref_hash)) for h in hashes) <= HASH_LIMIT]
        folder = target / ("parecidas_com_teste" if close else "") / cls
        folder.mkdir(parents=True, exist_ok=True)
        crop.save(folder / name, quality=95)
        if close:
            suspicious.append({"recorte": name, "parecida_com": close[:3]})

    report = {"nomes_de_classe_no_xml": dict(names_seen), "recortes_por_classe": dict(kept),
              "ignorados": dict(ignored), "parecidos_com_teste": suspicious}
    save_json(report, target.parent / f"{target.name}_info.json")

    print("\nNomes de classe encontrados nos .xml:", dict(names_seen))
    print("Ignorados (dime, outras classes ou caixas muito pequenas):", dict(ignored))
    print("\nRecortes por moeda:")
    for cls in CLASS_OF.values():
        print(f"  {cls:<32} {kept[cls]:>5}")
    print(f"\nRecortes muito parecidos com fotos de teste: {len(suspicious)} "
          f"(separados em {target / 'parecidas_com_teste'}; NÃO use no treino)")
    print(f"Recortes salvos em: {target}")
    print("Próximo passo: abra a pasta do Washington e apague os quarters de estados/parques (1999 em diante).")


if __name__ == "__main__":
    main()
