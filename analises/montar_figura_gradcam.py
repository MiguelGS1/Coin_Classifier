from pathlib import Path

from PIL import Image

# Junta, uma embaixo da outra, figuras de Grad-CAM já salvas (mesma largura), para a figura do artigo.
# Só lê as imagens; não roda nenhum modelo.
PASTA = Path("artifacts/experiments/gradcam_erros_teste")
FOTOS = [
    "gradcam_Jefferson Nickel 26505_177 Reverse.png",     # (a) em cápsula
    "gradcam_Washington Quarter 17345_41 Obverse.png",    # (b) tonalizado
    "gradcam_Washington Quarter 72403_66 Reverse.png",    # (c) luz alaranjada
]
SAIDA = PASTA / "gradcam_exemplos.png"

imagens = [Image.open(PASTA / nome).convert("RGB") for nome in FOTOS]
largura = max(img.width for img in imagens)
figura = Image.new("RGB", (largura, sum(img.height for img in imagens)), "white")
y = 0
for img in imagens:
    figura.paste(img, (0, y))
    y += img.height
figura.save(SAIDA)
print(f"Figura salva em: {SAIDA}")
