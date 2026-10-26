# Front do Classificador de Moedas

Interface local de demonstração do classificador. Ela usa os mesmos modelos e o mesmo pré-processamento avaliados no
artigo, importando os arquivos congelados (`prever_moeda_recortada.py`, `dataset_loader.py`, `config.py` e os três
`.keras`) sem alterar nenhum deles. Os resultados do trabalho estão no artigo; esta interface é só uma demonstração.

## Instalar

No mesmo ambiente Python usado no projeto (TensorFlow, Keras, numpy, Pillow e OpenCV já instalados), instale o Flask:

```
pip install "flask>=3,<4"
```

## Rodar

Na **pasta raiz do projeto** (a que tem o `config.py`):

```
python front/app.py
```

Espere aparecer `Pronto. Abra http://127.0.0.1:5000 no navegador.` e abra esse endereço. Para parar, `Ctrl+C` no
terminal.

O `app.py` já desliga as otimizações oneDNN (`TF_ENABLE_ONEDNN_OPTS=0`), como na medição de tempo do artigo.

## Como funciona

1. **Enviar foto:** aceita JPG e PNG (no iPhone: Ajustes › Câmera › Formatos › Mais Compatível).
2. **Conferir recorte:** só aparece quando o círculo da moeda não é encontrado ou encosta na borda da foto
   (a menos de 2% do lado). Essa regra é só do front e não muda nenhum parâmetro da detecção.
3. **Resultado:** moeda reconhecida e probabilidade de cada classe (média das três redes). A imagem mostrada é a que
   as redes receberam (224 × 224). Também é possível classificar com a foto inteira.

As fotos enviadas ficam numa pasta temporária, apagada quando o programa é fechado. Nada é enviado para a internet
(só as fontes da página vêm do Google Fonts; sem internet, a página usa a fonte do sistema).

O sistema sempre responde uma das três moedas, mesmo para outras moedas.
