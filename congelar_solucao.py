import argparse
import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import ARTIFACTS_DIR, BASE_DIR, DATASET_DIR, IMAGE_SIZE

# "Congela" a solução final: registra, num arquivo, a "impressão digital" (hash SHA-256) dos 3 modelos e dos scripts
# usados na avaliação, além do pré-processamento, da ordem das classes, da regra do ensemble e das versões
# (Python, TensorFlow). Também confere o resultados.json de cada modelo (semente, 40 épocas, sem EarlyStopping).
#   Sem --verificar: cria o registro (faça UMA vez, agora).
#   Com --verificar: recalcula tudo e compara com o registro (faça antes da avaliação final). Se qualquer arquivo
#   tiver mudado, avisa. Não altera modelos nem dados.
EXPERIMENTS = ARTIFACTS_DIR / "experiments"
MODELOS = {
    "s42": EXPERIMENTS / "divisao_series_v2_profundo_balanco_branco_cutout" / "modelo_divisao_series.keras",
    "s1": EXPERIMENTS / "divisao_series_v2_profundo_balanco_branco_cutout_40ep_semente1" / "modelo_divisao_series.keras",
    "s2": EXPERIMENTS / "divisao_series_v2_profundo_balanco_branco_cutout_40ep_semente2" / "modelo_divisao_series.keras",
}
SCRIPTS = ["config.py", "dataset_loader.py", "metrics.py", "io_utils.py", "make_group_split.py",
           "train_group_split_profundo_balanco_branco_cutout2.py", "avaliar_ensemble.py", "prever_ensemble.py"]
REGISTRO = ARTIFACTS_DIR / "protocolo_congelado"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Registra (ou verifica) a solução final congelada.")
    parser.add_argument("--verificar", action="store_true", help="Compara com o registro já salvo, sem alterar nada")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_record() -> dict:
    class_names = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())
    modelos = {}
    for nome, path in MODELOS.items():
        if not path.exists():
            raise SystemExit(f"Modelo não encontrado: {path}")
        results = json.loads((path.parent / "resultados.json").read_text(encoding="utf-8"))
        model = tf.keras.models.load_model(path)
        modelos[nome] = {
            "arquivo": str(path.relative_to(BASE_DIR)),
            "sha256": sha256(path),
            "semente": results.get("semente"),
            "sem_parada": results.get("sem_parada"),
            "epocas_executadas": results.get("epocas_executadas"),
            "melhor_epoca": results.get("melhor_epoca"),
            "observacao": observacao(results),
            "acuracia_teste_v2": results["teste_series_novas"]["acuracia"],
            "saidas": int(model.output_shape[-1]),
            "parametros": int(model.count_params()),
        }
    scripts = {s: sha256(BASE_DIR / s) for s in SCRIPTS if (BASE_DIR / s).exists()}
    return {
        "solucao": "Ensemble de 3 CNNs (mesma arquitetura, sementes 42, 1 e 2) por soft voting",
        "regra_ensemble": "média aritmética simples das probabilidades das 3 redes, sem pesos; classe = maior média",
        "ordem_das_classes": class_names,
        "pre_processamento": f"imagem RGB redimensionada para {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} (bilinear, sem manter a "
                             "proporção), valores 0-255 em float32 (dataset_loader.build_single_image_tensor)",
        "pesos_salvos": "estado da última época (época 40), sem checkpoint; ver a observação de cada modelo",
        "metricas": ["acurácia", "F1 macro", "matriz de confusão", "tempo por imagem"],
        "modelos": modelos,
        "scripts": scripts,
        "ambiente": {"python": platform.python_version(), "tensorflow": tf.__version__, "numpy": np.__version__,
                     "sistema": f"{platform.system()} {platform.release()}"},
    }


def observacao(results: dict) -> str:
    # O modelo s42 veio do script antigo (com EarlyStopping), que não grava "sem_parada". Ele vale como "40 épocas,
    # pesos da última" só se o treino foi até a época 40 E a melhor época (menor perda na validação) foi a 40: aí o
    # EarlyStopping nunca interrompeu, e restaurar os "melhores pesos" dá os mesmos pesos da última época.
    if results.get("sem_parada") is True:
        return "treinado com --sem_parada (40 épocas fixas, pesos da última época)"
    if results.get("sem_parada") is None and results.get("epocas_executadas") == 40 and results.get("melhor_epoca") == 40:
        return ("treinado com EarlyStopping (paciência 5, restore_best_weights), que não interrompeu o treino; "
                "a melhor época foi a 40, então os pesos salvos são os da última época, como nos demais")
    return "PROTOCOLO DIFERENTE"


def problems(record: dict) -> list:
    issues = []
    for nome, m in record["modelos"].items():
        if m["observacao"] == "PROTOCOLO DIFERENTE" or m["epocas_executadas"] != 40:
            issues.append(f"{nome}: sem_parada={m['sem_parada']}, épocas={m['epocas_executadas']}, "
                          f"melhor época={m['melhor_epoca']} (esperado: 40 épocas e pesos da última)")
        if m["saidas"] != len(record["ordem_das_classes"]):
            issues.append(f"{nome}: o modelo tem {m['saidas']} saídas, mas há {len(record['ordem_das_classes'])} classes")
    return issues


def main() -> None:
    args = parse_args()
    json_path = REGISTRO / "protocolo_congelado.json"
    record = build_record()

    if args.verificar:
        if not json_path.exists():
            raise SystemExit(f"Não existe registro em {json_path}. Rode primeiro sem --verificar.")
        saved = json.loads(json_path.read_text(encoding="utf-8"))
        changed = [f"modelo {n}" for n in saved["modelos"] if saved["modelos"][n]["sha256"] != record["modelos"][n]["sha256"]]
        changed += [f"script {s}" for s in saved["scripts"] if saved["scripts"][s] != record["scripts"].get(s)]
        if saved["ordem_das_classes"] != record["ordem_das_classes"]:
            changed.append("ordem das classes")
        if changed:
            print("ATENÇÃO, mudou desde o congelamento: " + ", ".join(changed))
        else:
            print(f"OK: modelos, scripts e ordem das classes iguais ao registro de {saved['data']}.")
        if saved["ambiente"] != record["ambiente"]:
            print(f"Aviso: o ambiente mudou ({saved['ambiente']} -> {record['ambiente']}).")
        return

    if json_path.exists():
        raise SystemExit(f"Já existe um registro em {json_path}. Para conferir, use --verificar.")
    issues = problems(record)
    if issues:
        print("NÃO congelado. Corrija antes:\n  " + "\n  ".join(issues))
        return
    record["data"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    REGISTRO.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# Solução final congelada ({record['data']})", "", f"- **Solução:** {record['solucao']}",
             f"- **Regra do ensemble:** {record['regra_ensemble']}",
             f"- **Ordem das classes:** {', '.join(record['ordem_das_classes'])}",
             f"- **Pré-processamento:** {record['pre_processamento']}", f"- **Pesos salvos:** {record['pesos_salvos']}",
             f"- **Métricas:** {', '.join(record['metricas'])}",
             f"- **Ambiente:** Python {record['ambiente']['python']}, TensorFlow {record['ambiente']['tensorflow']}, "
             f"{record['ambiente']['sistema']}", "", "## Modelos", "",
             "| Modelo | Semente | Épocas | Teste v2 | SHA-256 |", "|---|---|---|---|---|"]
    for nome, m in record["modelos"].items():
        lines.append(f"| {nome} | {m['semente']} | {m['epocas_executadas']} | {m['acuracia_teste_v2'] * 100:.1f}% | "
                     f"`{m['sha256']}` |")
    lines += [""] + [f"- **{nome}:** {m['observacao']}" for nome, m in record["modelos"].items()]
    lines += ["", "## Scripts", "", "| Arquivo | SHA-256 |", "|---|---|"]
    lines += [f"| {s} | `{h}` |" for s, h in record["scripts"].items()]
    (REGISTRO / "PROTOCOLO_CONGELADO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nRegistro salvo em: {REGISTRO} (protocolo_congelado.json e PROTOCOLO_CONGELADO.md)")


if __name__ == "__main__":
    main()
