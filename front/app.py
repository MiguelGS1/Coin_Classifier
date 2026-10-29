import atexit
import os
import shutil
import sys
import tempfile
import threading
import uuid
from pathlib import Path

# Sem isso, a inferência fica cerca de 100 vezes mais lenta neste processador (Seção 5.7 do artigo).
# Precisa ser definido antes de importar o TensorFlow.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
from flask import Flask, abort, redirect, render_template, request, send_file, url_for
from PIL import Image, ImageDraw

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))  # para importar os módulos congelados da raiz do projeto

import tensorflow as tf  # noqa: E402

import prever_moeda_recortada as recorte  # noqa: E402  (congelado: só importado, nunca alterado)
from config import IMAGE_SIZE  # noqa: E402
from dataset_loader import build_single_image_tensor  # noqa: E402

# Front de demonstração do classificador. Usa exatamente o pré-processamento e os modelos congelados:
# sem recorte, build_single_image_tensor; com recorte, load_rgb + isolate_coin(fundo branco) + resize 224 x 224.
# A resposta é a média das probabilidades das três redes. Não treina nada e não altera nenhum arquivo do projeto.

# Ordem das saídas das redes (fixa; não é lida da pasta dataset/, que pode não existir neste computador)
CLASSES = ["Jefferson Nickels, 1938-Date", "Lincoln Cents, 1909-Date", "Washington Quarters, 1932-1998"]
MOEDAS = {
    "Jefferson Nickels, 1938-Date": {"nome": "Jefferson Nickel", "valor": "5 centavos", "periodo": "1938–2003",
                                     "cor": "#5F6E7A"},
    "Lincoln Cents, 1909-Date": {"nome": "Lincoln Cent", "valor": "1 centavo", "periodo": "1909–2008",
                                 "cor": "#A85A2A"},
    "Washington Quarters, 1932-1998": {"nome": "Washington Quarter", "valor": "25 centavos",
                                       "periodo": "1932–1998", "cor": "#A3AFBA"},
}
EXTENSOES = {".jpg", ".jpeg", ".png"}
BORDA_SUSPEITA = 0.02  # círculo a menos de 2% do lado da foto da borda: pedir conferência (regra só do front)
LADO_PREVIA = 900      # maior lado da foto mostrada na tela de conferência

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024

PASTA_TEMP = Path(tempfile.mkdtemp(prefix="classificador_moedas_"))
atexit.register(shutil.rmtree, PASTA_TEMP, ignore_errors=True)

print("Carregando as três redes...")
MODELOS = [tf.keras.models.load_model(m) for m in recorte.DEFAULT_MODELS]
print("Pronto. Abra http://127.0.0.1:5000 no navegador.")

TRAVA = threading.Lock()
FOTOS = {}  # id -> dados da foto enviada


def prever(tensor) -> np.ndarray:
    with TRAVA:
        probs = [modelo(tensor, training=False).numpy()[0] for modelo in MODELOS]
    return np.mean(probs, axis=0)


def tensor_recorte(coin: np.ndarray):
    return tf.image.resize(tf.constant(coin[None], dtype=tf.float32), IMAGE_SIZE)


def salvar_png(tensor, caminho: Path) -> None:
    pixels = np.clip(np.round(tensor.numpy()[0]), 0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(caminho)


def circulo_suspeito(circulo, largura: int, altura: int) -> bool:
    cx, cy, r = circulo
    return (cx - r < BORDA_SUSPEITA * largura or cx + r > (1 - BORDA_SUSPEITA) * largura
            or cy - r < BORDA_SUSPEITA * altura or cy + r > (1 - BORDA_SUSPEITA) * altura)


def desenhar_previa(rgb: np.ndarray, circulo, caminho: Path) -> None:
    # Foto enviada com o círculo encontrado (azul) e o quadrado recortado (branco)
    altura, largura = rgb.shape[:2]
    escala = min(1.0, LADO_PREVIA / max(altura, largura))
    img = Image.fromarray(rgb).resize((round(largura * escala), round(altura * escala)))
    desenho = ImageDraw.Draw(img)
    espessura = max(3, round(min(img.size) / 150))
    if circulo is None:
        lado = min(img.size)
        x0, y0 = (img.width - lado) // 2, (img.height - lado) // 2
        desenho.rectangle([x0, y0, x0 + lado, y0 + lado], outline="#FFFFFF", width=espessura)
    else:
        cx, cy, r = (v * escala for v in circulo)
        meio = recorte.MARGIN * r
        desenho.rectangle([cx - meio, cy - meio, cx + meio, cy + meio], outline="#FFFFFF", width=max(2, espessura // 2))
        desenho.ellipse([cx - r, cy - r, cx + r, cy + r], outline="#2D5A7B", width=espessura)
    img.save(caminho)


def foto_ou_404(id_foto: str) -> dict:
    foto = FOTOS.get(id_foto)
    if foto is None:
        abort(404)
    return foto


def classificar(foto: dict, modo: str) -> None:
    if modo in foto["resultados"]:
        return
    pasta = foto["pasta"]
    if modo == "recorte":
        coin, _ = recorte.isolate_coin(recorte.load_rgb(str(foto["arquivo"])), 255)
        tensor = tensor_recorte(coin)
    else:
        # Igual à avaliação: lê o arquivo, 3 canais, 224 x 224, sem aplicar a rotação EXIF
        tensor = build_single_image_tensor(str(foto["arquivo"]))
    salvar_png(tensor, pasta / f"rede_{modo}.png")
    foto["resultados"][modo] = prever(tensor)


@app.get("/")
def enviar():
    return render_template("enviar.html", etapa=1, erro=request.args.get("erro"), moedas=[MOEDAS[c] for c in CLASSES])


@app.post("/enviar")
def receber():
    arquivo = request.files.get("foto")
    if arquivo is None or not arquivo.filename:
        return redirect(url_for("enviar", erro="Escolha uma foto antes de classificar."))
    extensao = Path(arquivo.filename).suffix.lower()
    if extensao not in EXTENSOES:
        return redirect(url_for("enviar", erro="Formato não aceito. Envie uma foto JPG ou PNG."))

    id_foto = uuid.uuid4().hex
    pasta = PASTA_TEMP / id_foto
    pasta.mkdir()
    caminho = pasta / f"original{extensao}"
    arquivo.save(caminho)
    try:
        rgb = recorte.load_rgb(str(caminho))
    except Exception:
        shutil.rmtree(pasta, ignore_errors=True)
        return redirect(url_for("enviar", erro="Não foi possível abrir a foto. Envie um arquivo JPG ou PNG válido."))

    altura, largura = rgb.shape[:2]
    circulo = recorte.find_coin(rgb)
    suspeita = circulo is None or circulo_suspeito(circulo, largura, altura)
    FOTOS[id_foto] = {"pasta": pasta, "arquivo": caminho, "nome": arquivo.filename, "encontrado": circulo is not None,
                      "suspeita": suspeita, "resultados": {}}

    # Sempre classifica com recorte; se o recorte for suspeito, o resultado mostra um aviso para conferir (etapa 3)
    classificar(FOTOS[id_foto], "recorte")
    return redirect(url_for("resultado", id_foto=id_foto, modo="recorte"))


@app.get("/conferir/<id_foto>")
def conferir(id_foto):
    foto = foto_ou_404(id_foto)
    if not (foto["pasta"] / "previa.png").exists():
        rgb = recorte.load_rgb(str(foto["arquivo"]))
        circulo = recorte.find_coin(rgb)
        desenhar_previa(rgb, circulo, foto["pasta"] / "previa.png")
    foto["conferido"] = True
    return render_template("conferir.html", etapa=3, conferido=True, id_foto=id_foto, foto=foto)


@app.post("/classificar/<id_foto>")
def classificar_com(id_foto):
    foto = foto_ou_404(id_foto)
    modo = request.form.get("modo")
    if modo not in ("recorte", "inteira"):
        abort(400)
    classificar(foto, modo)
    return redirect(url_for("resultado", id_foto=id_foto, modo=modo))


@app.get("/resultado/<id_foto>")
def resultado(id_foto):
    foto = foto_ou_404(id_foto)
    modo = request.args.get("modo", "recorte")
    if modo not in ("recorte", "inteira"):
        abort(400)
    classificar(foto, modo)
    probs = foto["resultados"][modo]
    ordem = np.argsort(probs)[::-1]
    barras = [{**MOEDAS[CLASSES[i]], "prob": float(probs[i])} for i in ordem]
    return render_template("resultado.html", etapa=2, conferido=foto.get("conferido", False), id_foto=id_foto,
                           foto=foto, modo=modo, vencedora=barras[0], barras=barras)


@app.get("/imagem/<id_foto>/<tipo>")
def imagem(id_foto, tipo):
    foto = foto_ou_404(id_foto)
    arquivos = {"previa": "previa.png", "recorte": "rede_recorte.png", "inteira": "rede_inteira.png"}
    if tipo not in arquivos or not (foto["pasta"] / arquivos[tipo]).exists():
        abort(404)
    return send_file(foto["pasta"] / arquivos[tipo], mimetype="image/png", max_age=0)


@app.template_filter("pct")
def pct(valor: float) -> str:
    return f"{valor * 100:.1f}".replace(".", ",") + "%"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
