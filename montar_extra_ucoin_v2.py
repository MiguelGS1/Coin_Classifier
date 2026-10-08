import argparse
import csv
import hashlib
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from config import BASE_DIR

# Monta o dataset/extra_ucoin_limpov2 DO ZERO, direto do World Coins (wanderdust, Kaggle), sem usar o controle_eua.
#  1. Pega as fotos das pastas 206 (Lincoln), 207 (Jefferson) e 209 (Washington), em train, validation e test.
#  2. Tira as fotos listadas no arquivo de exclusões (decididas olhando as fotos, cada uma com o motivo).
#  3. Tira as cópias exatas (MD5): dentro do próprio conjunto (fica a primeira) e as que existem no dataset/raw.
#  4. Copia o resto para a pasta de saída e salva o manifesto (todas as fotos candidatas e o que aconteceu com cada
#     uma) e as folhas de revisão (fotos numeradas, para decidir as exclusões).
# Só lê o World Coins e o raw; não altera nenhuma imagem.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
WORLD_COINS_FOLDERS = {  # pasta do World Coins -> classe do projeto
    "206": "Lincoln Cents, 1909-Date",
    "207": "Jefferson Nickels, 1938-Date",
    "209": "Washington Quarters, 1932-1998",
}
SPLITS = ("train", "validation", "test")  # divisões do próprio World Coins (para nós, tudo é teste)
HASH_LIMIT = 3  # distância máxima do hash visual (0 a 64) para avisar que uma foto é quase igual a uma do treino


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monta o extra_ucoin_limpov2 a partir do World Coins.")
    parser.add_argument("--origem", required=True,
                        help="Pasta do World Coins que contém train, validation e test (ex.: Downloads/coins/data)")
    parser.add_argument("--exclusoes", default=str(BASE_DIR / "exclusoes_extra_ucoin_v2.csv"),
                        help="CSV (separado por ;) com as colunas divisao;pasta;arquivo;motivo")
    parser.add_argument("--comparar", default=str(BASE_DIR / "dataset" / "raw"),
                        help="Pasta do treino para procurar cópias (padrão: dataset/raw)")
    parser.add_argument("--saida", default=str(BASE_DIR / "dataset" / "extra_ucoin_limpov2"))
    return parser.parse_args()


def fingerprints(path: Path) -> tuple:
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    small = np.asarray(Image.open(path).convert("L").resize((9, 8)), dtype=np.int16)
    return md5, (small[:, 1:] > small[:, :-1]).flatten()


def read_exclusions(path: Path) -> dict:
    if not path.exists():
        print(f"Arquivo de exclusões {path} não encontrado: nenhuma foto será excluída à mão.")
        return {}
    with open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    return {(r["divisao"].strip(), r["pasta"].strip(), r["arquivo"].strip()): r["motivo"].strip() for r in rows}


def review_sheet(items: list, title: str, path: Path) -> None:
    # Folha com todas as fotos candidatas de uma moeda, cada uma com número, divisão e nome
    cols, size, label = 6, 260, 34
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * size, rows * (size + label) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 8), title, fill="black")
    for i, item in enumerate(items):
        image = Image.open(item["caminho"]).convert("RGB")
        image.thumbnail((size, size))
        x, y = (i % cols) * size, 30 + (i // cols) * (size + label)
        sheet.paste(image, (x, y + label))
        color = "black" if item["situacao"] == "usada" else "red"
        draw.text((x + 3, y + 2), f"#{i + 1} {item['divisao']}/{item['arquivo'][:26]}", fill=color)
        draw.text((x + 3, y + 16), item["situacao"] if item["situacao"] == "usada" else "EXCLUIDA", fill=color)  # sem acento: a fonte padrão do PIL não tem
    sheet.save(path, quality=85)


def main() -> None:
    args = parse_args()
    origin, out_dir = Path(args.origem), Path(args.saida)
    if out_dir.exists():
        raise SystemExit(f"A pasta {out_dir} já existe. Apague ou use outra --saida.")
    exclusions = read_exclusions(Path(args.exclusoes))

    # 1) Candidatas
    items = []
    for folder, cls in WORLD_COINS_FOLDERS.items():
        for split in SPLITS:
            source = origin / split / folder
            if not source.exists():
                raise SystemExit(f"Pasta {source} não encontrada. Confira o --origem.")
            for p in sorted(source.iterdir()):
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    md5, dhash = fingerprints(p)
                    items.append({"classe": cls, "divisao": split, "pasta": folder, "arquivo": p.name, "caminho": p,
                                  "md5": md5, "dhash": dhash, "situacao": "usada", "motivo": "", "aviso": ""})
    print(f"Fotos candidatas no World Coins: {len(items)}")

    # 2) Exclusões à mão (cada nome do CSV tem que existir, para não passar erro de digitação)
    found = {(it["divisao"], it["pasta"], it["arquivo"]) for it in items}
    missing = [k for k in exclusions if k not in found]
    if missing:
        raise SystemExit("Estas linhas do arquivo de exclusões não correspondem a nenhuma foto:\n  "
                         + "\n  ".join("/".join(k) for k in missing))
    for it in items:
        key = (it["divisao"], it["pasta"], it["arquivo"])
        if key in exclusions:
            it["situacao"], it["motivo"] = "excluida", f"à mão: {exclusions[key]}"

    # 3) Cópias exatas: no treino (dataset/raw) e dentro do próprio conjunto
    raw_files = [p for p in Path(args.comparar).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    raw = [(p, *fingerprints(p)) for p in raw_files]
    raw_md5 = {md5: p for p, md5, _ in raw}
    raw_hashes = np.stack([h for _, _, h in raw]) if raw else np.zeros((0, 64), bool)
    print(f"Fotos do treino comparadas ({args.comparar}): {len(raw)}")
    first_copy = {}
    for it in items:
        if it["situacao"] != "usada":
            continue
        if it["md5"] in raw_md5:
            it["situacao"], it["motivo"] = "excluida", f"cópia exata do treino: {raw_md5[it['md5']].name}"
        elif it["md5"] in first_copy:
            it["situacao"], it["motivo"] = "excluida", f"cópia exata de {first_copy[it['md5']]}"
        else:
            first_copy[it["md5"]] = f"{it['divisao']}/{it['pasta']}/{it['arquivo']}"
    # Fotos QUASE iguais às do treino: só avisa, quem decide é você. (Dentro do próprio conjunto não dá para usar o
    # hash visual: as fotos do uCoin são tão padronizadas que moedas diferentes dão hashes quase iguais.)
    used = [it for it in items if it["situacao"] == "usada"]
    for it in used:
        if len(raw_hashes):
            dist = (raw_hashes != it["dhash"]).sum(axis=1)
            if dist.min() <= HASH_LIMIT:
                it["aviso"] = f"quase igual a {raw[int(dist.argmin())][0].name} do treino (distância {int(dist.min())})"

    # 4) Cópia, manifesto e folhas de revisão
    info_dir = out_dir.parent / f"{out_dir.name}_info"
    info_dir.mkdir(parents=True, exist_ok=True)
    for it in used:
        target = out_dir / it["classe"] / f"{it['divisao']}_{it['arquivo']}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(it["caminho"], target)
    with open(info_dir / "manifesto.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["classe", "divisao", "pasta", "arquivo", "md5", "situacao", "motivo", "aviso"])
        for it in items:
            writer.writerow([it[k] for k in ("classe", "divisao", "pasta", "arquivo", "md5", "situacao", "motivo", "aviso")])
    for cls in WORLD_COINS_FOLDERS.values():
        cls_items = [it for it in items if it["classe"] == cls]
        review_sheet(cls_items, f"{cls} - todas as candidatas (vermelho = excluida)",
                     info_dir / f"revisao_{cls.split()[0]}.jpg")

    print(f"\n{'moeda':<34}{'candidatas':>11}{'à mão':>8}{'cópias':>8}{'usadas':>8}")
    for cls in WORLD_COINS_FOLDERS.values():
        c = [it for it in items if it["classe"] == cls]
        hand = sum(it["motivo"].startswith("à mão") for it in c)
        copies = sum(it["motivo"].startswith("cópia") for it in c)
        print(f"{cls:<34}{len(c):>11}{hand:>8}{copies:>8}{sum(it['situacao'] == 'usada' for it in c):>8}")
    warnings = [it for it in used if it["aviso"]]
    print(f"\nFotos usadas com AVISO de quase igual ao treino: {len(warnings)} (abra os pares e decida; se for a mesma "
          "foto, coloque no arquivo de exclusões e rode de novo)")
    for it in warnings:
        print(f"  {it['divisao']}/{it['pasta']}/{it['arquivo']}: {it['aviso']}")
    print(f"\nDataset salvo em: {out_dir}")
    print(f"Manifesto e folhas de revisão em: {info_dir}")


if __name__ == "__main__":
    main()
