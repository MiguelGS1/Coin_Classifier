# Classificador de moedas com classe `outra_moeda`

Este projeto foi feito para classificar moedas **conhecidas** do seu dataset e também estimar a probabilidade de a imagem pertencer a **uma moeda que não está no dataset**.

## Como a ideia funciona

O modelo aprende as classes conhecidas com um classificador multiclasse normal. Depois, no momento da previsão, ele combina dois sinais:

1. **Confiança do softmax**: se a confiança máxima for baixa, aumenta a chance de ser `outra_moeda`.
2. **Distância no espaço de características**: o projeto calcula um vetor de características da imagem e mede quão distante ele está do centro visual das classes conhecidas. Se ficar longe demais, aumenta a chance de ser `outra_moeda`.

Isso é uma abordagem de **open-set recognition** simples e expansível.

## Estrutura esperada do dataset

```text
dataset/
  raw/
    1_real_brasil/
    1_euro/
    10_ore_noruega/
    5_pesos_mexico/
```

Cada pasta é uma classe conhecida. Coloque **somente imagens verdadeiras** de cada moeda listada.

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Treino

```bash
python train.py
```

Arquivos gerados:

- `artifacts/coin_classifier.keras`
- `artifacts/class_names.json`
- `artifacts/class_centers.json`
- `artifacts/openset_calibration.json`
- `artifacts/training_history.json`

## Previsão

```bash
python predict.py --image caminho/para/imagem.jpg
```

Exemplo de saída:

```text
Classe mais provável: 1_real_brasil
Confiança: 54.20%

Distribuição das probabilidades:
- 1_real_brasil: 54.20%
- 1_euro: 17.30%
- 10_ore_noruega: 9.10%
- outra_moeda: 19.40%
```

## Avaliação

```bash
python evaluate.py
```

## Onde mexer depois

### Ajustar hiperparâmetros
Edite `config.py`:
- tamanho da imagem
- batch size
- épocas
- pesos da classe `outra_moeda`
- percentis de calibração

### Trocar o modelo
Edite `model_builder.py`.

### Mudar a lógica de `outra_moeda`
Edite `openset_utils.py`.

## Limitação importante

A classe `outra_moeda` não é aprendida com exemplos reais por padrão. Ela é uma **estimativa de desconhecido** baseada em baixa confiança + distância visual. Isso funciona bem como ponto de partida, mas melhora bastante se você futuramente adicionar um conjunto de moedas externas só para calibrar a detecção de desconhecidos.


## Dataset inicial incluído neste pacote

Este pacote já vem com um dataset inicial mínimo em:

```text
dataset/raw/1_real_brasil/
dataset/raw/1_euro/
```

Imagens incluídas:
- `1_real_brasil/1_real_brasil_2019.png`
- `1_euro/1_euro_vitruvian.png`
- `1_euro/1_euro_mapa.png`

Como esse dataset é muito pequeno, use-o só para validar o fluxo do código. Para um resultado melhor, adicione mais fotos reais de cada moeda e treine novamente.

## para treinar
python3 train.py

## para iniciar
source .venv/bin/activate
python3 predict.py --image "dataset/raw/Jefferson Nickels, 1938-Date/Jefferson Nickel 26324_109 Reverse.jpg"