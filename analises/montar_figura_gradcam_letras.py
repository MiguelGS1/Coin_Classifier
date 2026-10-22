from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Igual ao montar_figura_gradcam.py (junta, uma embaixo da outra, as figuras de Grad-CAM já salvas), mas escreve
# as letras (a), (b) e (c) numa margem à esquerda de cada linha, como citado na legenda do artigo.
# Só lê as imagens; não roda nenhum modelo.
PASTA = Path("artifacts/experiments/gradcam_erros_teste")
FOTOS = [
    "gradcam_Jefferson Nickel 26505_177 Reverse.png",     # (a) em cápsula
    "gradcam_Washington Quarter 17345_41 Obverse.png",    # (b) tonalizado
    "gradcam_Washington Quarter 72403_66 Reverse.png",    # (c) luz alaranjada
]
LETRAS = ["(a)", "(b)", "(c)"]
SAIDA = PASTA / "gradcam_exemplos_letras.png"
MARGEM = 90


def fonte(tamanho: int) -> ImageFont.ImageFont:
    for nome in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(nome, tamanho)
        except OSError:
            pass
    return ImageFont.load_default(size=tamanho)


imagens = [Image.open(PASTA / nome).convert("RGB") for nome in FOTOS]
largura = max(img.width for img in imagens) + MARGEM
figura = Image.new("RGB", (largura, sum(img.height for img in imagens)), "white")
desenho = ImageDraw.Draw(figura)
letra_fonte = fonte(40)
y = 0
for img, letra in zip(imagens, LETRAS):
    figura.paste(img, (MARGEM, y))
    desenho.text((MARGEM // 2, y + img.height // 2), letra, fill="black", font=letra_fonte, anchor="mm")
    y += img.height
figura.save(SAIDA)
print(f"Figura salva em: {SAIDA}")
