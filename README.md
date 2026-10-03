# Coin_Classifier

Classificador de moedas americanas com uma rede neural convolucional (CNN) treinada do zero, em TensorFlow/Keras.
O projeto faz parte do TCC *"Atalhos de aprendizado e generalização fora do domínio em redes convolucionais:
um estudo de caso com moedas"* (Ciência da Computação, PUCRS).

O modelo classifica três moedas:

- **Jefferson Nickels** (5 centavos, 1938 até hoje)
- **Lincoln Cents** (1 centavo, 1909 até hoje)
- **Washington Quarters** (25 centavos, 1932–1998)

Além do treino e da previsão, o repositório tem os scripts dos experimentos do TCC: validação com fotos de
outras fontes, fotos de celular, divisão do dataset por série, testes de atalhos (oclusão e cor) e aumento de dados.

---

## 1. Baixar o projeto

Clone o repositório:

```bash
git clone https://github.com/MiguelGS1/Coin_Classifier.git
cd Coin_Classifier
```

Ou baixe o .zip pelo GitHub e abra a pasta do projeto no terminal.

## 2. Criar e ativar o ambiente virtual

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Para sair do ambiente virtual, use `deactivate`.

## 3. Instalar as dependências

Linux/macOS:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Windows:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Alguns experimentos usam também o OpenCV:

```bash
python -m pip install opencv-python
```

## 4. Baixar e organizar o dataset

O dataset não está incluído no repositório por causa do tamanho (as pastas `dataset/` e `artifacts/` estão no
`.gitignore`).

Baixe o **U.S. Coins** em:

https://www.kaggle.com/datasets/sergiosaharovskiy/uscoins

Coloque as imagens dentro de `dataset/raw/`. A estrutura deve ficar assim:

```
dataset/raw/
  Jefferson Nickels, 1938-Date/
  Lincoln Cents, 1909-Date/
  Washington Quarters, 1932-1998/
```

Cada pasta é uma classe. São 3.749 imagens no total (Jefferson 1.214, Lincoln 1.285, Washington 1.250).

**Importante:** os scripts nunca alteram `dataset/raw/`. Quando um experimento precisa de outra organização das
imagens, ele cria uma cópia em outra pasta.

## 5. Treinar o modelo

```bash
python train.py
```

O treino usa a divisão aleatória por imagem (80% treino / 20% validação, semente 42), Adam com taxa de
aprendizado 1e-3, lotes de 16 imagens, até 40 épocas e parada antecipada (EarlyStopping, paciência 5).

Ao final, é criada a pasta `artifacts/`, com o modelo treinado (`coin_classifier.keras`), os nomes das classes e
os arquivos usados na previsão.

### Quando rodar `train.py` novamente?

Apenas se:

- adicionar ou remover imagens do dataset;
- adicionar ou remover uma classe de moeda;
- apagar a pasta `artifacts/`;
- alterar configurações do modelo (`config.py`, `model_builder.py`).

## 6. Testar uma imagem

```bash
python predict.py --image "caminho/da/imagem.jpg"
```

Exemplo com uma imagem do dataset:

```bash
python predict.py --image "dataset/raw/Jefferson Nickels, 1938-Date/Jefferson Nickel 26324_109 Reverse.jpg"
```

O programa mostra a porcentagem de cada classe de moeda e também de `outra_moeda`.

Para usar de novo depois, basta ativar o ambiente virtual e rodar o `predict.py`; não é preciso treinar de novo.

## 7. Avaliar o modelo

```bash
python evaluate.py   # acurácia na validação, considerando também outra_moeda
python metrics.py    # acurácia, precisão, revocação, F1, matriz de confusão e curvas de treino
```

O `metrics.py` salva os resultados (figuras e tabela em LaTeX) em `artifacts/experiments/`.

---

## 8. Experimentos do TCC

Todos os experimentos **só leem** os dados e o modelo existentes e salvam os resultados em pastas novas, dentro de
`artifacts/experiments/`. Se a pasta de resultado já existir, o script para e avisa, para não sobrescrever nada.

### 8.1 Divisão por série (teste com séries nunca vistas)

As fotos do U.S. Coins vêm em **séries** (fotos de uma mesma origem, com o mesmo fundo, luz e enquadramento).
O número no nome do arquivo identifica a série: `Lincoln Cent 37633_208 Obverse.jpg` → série `37633`.
Na divisão aleatória por imagem, quase toda a validação tem fotos da mesma série no treino, o que deixa o
resultado otimista. Este experimento separa o teste por **série inteira**:

```
dataset/raw (3.749 fotos)
│
│  ETAPA 1 - divisão por série (make_group_split.py)
│  Séries inteiras são sorteadas; cada série vai inteira para um lado só.
│
├── ~85% → treino_validacao
│     │
│     │  ETAPA 2 - divisão por foto (train_group_split.py)
│     │  Mesma divisão do projeto original (80/20), dentro de treino_validacao.
│     │
│     ├── 80% → treino
│     └── 20% → validação (fotos diferentes, mas das mesmas séries do treino)
│
└── ~15% → teste (séries que não aparecem no treino nem na validação)
```

Regras usadas para formar as séries:

- séries que têm arquivos idênticos entre si (mesma imagem com nomes diferentes) são unidas, para que cópias
  nunca fiquem uma no treino e outra no teste;
- as imagens de Washington com dois números no nome (ex.: `Washington Quarter 11 6_2.jpg`) formam uma única
  coleção de fotos, com o mesmo estilo, e por isso são tratadas como uma série só.

Comandos:

```bash
# Etapa 1: cria dataset/split_series_v2 (treino_validacao e teste) e divisao.json
python make_group_split.py

# Etapa 2: treina e avalia na validação (séries vistas) e no teste (séries nunca vistas)
python train_group_split.py --pasta dataset/split_series_v2 --saida artifacts/experiments/divisao_series_v2
```

Resultados em `artifacts/experiments/divisao_series_v2/` (métricas, matriz de confusão, curvas de treino, tabela
em LaTeX e o modelo).

### 8.2 Fotos de outras fontes

| Script | O que faz |
|---|---|
| `controle_modelo_original.py` | Avalia o modelo em `dataset/controle_eua` (fotos de moedas americanas do dataset World Coins, de outro site) |
| `testar_celular.py` | Avalia um modelo na validação e em `dataset/celular/teste_celular` (620 fotos de celular, do projeto *us-currency-recognition*); use `--modelo` e `--nome` |
| `checar_duplicatas.py` | Procura arquivos idênticos entre treino e validação |

### 8.3 Atalhos e aumento de dados

| Script | O que faz |
|---|---|
| `occlusion_test.py` | Cobre o centro, a borda ou o fundo da moeda e converte para tons de cinza, para ver em que o modelo se apoia |
| `treinar_aumento.py` | Treina variantes com outro aumento de dados (`--variante sem_aumento`, `rotacao`, `luz_cor`, `forte`) e testa no celular |
| `degradacoes_validacao.py` | Aplica desfoque, luz amarelada, escurecimento e baixa resolução às imagens de validação |
| `fatores_celular.py` | Mede nitidez, brilho e tom das fotos de celular e relaciona com o acerto |

O guia da parte de degradações está em `GUIA_MIGUEL_degradacoes.md`.

---

## 9. Estrutura do projeto

| Arquivo | Função |
|---|---|
| `config.py` | Caminhos e configurações (tamanho da imagem, lote, semente, divisão) |
| `dataset_loader.py` | Carrega o dataset e faz a divisão treino/validação |
| `model_builder.py` | Arquitetura da CNN (4 blocos convolucionais, 32–256 filtros, ~422 mil parâmetros) |
| `train.py` | Treina o modelo |
| `evaluate.py` / `metrics.py` | Métricas e matriz de confusão |
| `predict.py` | Classifica uma imagem |
| `openset_utils.py` | Cálculo da probabilidade de `outra_moeda` |
| `io_utils.py` | Leitura e escrita de arquivos JSON |

## 10. Dados externos usados nos experimentos

| Pasta | Origem |
|---|---|
| `dataset/raw` | U.S. Coins – Sergio Saharovskiy, Kaggle (2023) |
| `dataset/controle_eua` | World Coins – wanderdust, Kaggle (2019), pastas de 1 e 5 centavos americanos, conferidas à mão |
| `dataset/celular/teste_celular` | us-currency-recognition (UNC Charlotte), GitHub `faustogerman/us-currency-recognition`, licença Apache-2.0 |

Esses dados não estão no repositório.
