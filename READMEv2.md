# Coin_Classifier

Classificador de moedas americanas com uma rede neural convolucional (CNN) treinada do zero, em TensorFlow/Keras.
O projeto faz parte do TCC *"Atalhos de aprendizado e generalização fora do domínio em redes convolucionais:
um estudo de caso com moedas"* (Ciência da Computação, PUCRS).

O modelo classifica três moedas:

- **Jefferson Nickels** (5 centavos, 1938 até hoje)
- **Lincoln Cents** (1 centavo, 1909 até hoje)
- **Washington Quarters** (25 centavos, 1932–1998)

O repositório tem duas versões do fluxo de trabalho:

| | Versão original | Versão atual (divisão por série) |
|---|---|---|
| Divisão dos dados | Aleatória por imagem (80/20) | Por **série**: o teste só tem séries que o modelo nunca viu |
| Modelo | 4 blocos convolucionais (~422 mil parâmetros) | **6 blocos convolucionais** (~1,6 milhão de parâmetros) |
| Dados de treino | U.S. Coins | U.S. Coins (opcional: + dataset LPR) |
| Scripts principais | `train.py`, `predict.py`, `evaluate.py`, `metrics.py` | `make_group_split.py`, `train_group_split_profundo.py` |
| Seções deste README | [5](#5-versão-original-divisão-aleatória-por-imagem) | [6](#6-versão-atual-divisão-por-série--modelo-com-6-blocos) e [7](#7-opcional-treinar-com-o-dataset-lpr) |

---

## Ordem para rodar (resumo)

**Versão atual** (o que foi usado nos resultados finais):

```
1. Preparar o ambiente                       (seções 1 a 3)
2. Colocar o U.S. Coins em dataset/raw       (seção 4)
3. python make_group_split.py                → cria dataset/split_series_v2
4. python train_group_split_profundo.py      → treina o modelo de 6 blocos e avalia no teste
5. (opcional) LPR: baixar → recortar_lpr.py → montar_v2_extra.py → treinar com --pasta dataset/split_series_v2_lpr
6. (opcional) análises: oclusao_v2.py, gradcam_v2.py, avaliar_pasta.py
```

**Versão original:**

```
1. Preparar o ambiente e o dataset/raw       (seções 1 a 4)
2. python train.py                            → treina o modelo de 4 blocos
3. python predict.py --image ...              → classifica uma imagem
```

---

## 1. Baixar o projeto

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

## 4. Baixar e organizar o dataset principal (U.S. Coins)

O dataset não está incluído no repositório por causa do tamanho (as pastas `dataset/` e `artifacts/` estão no
`.gitignore`).

Baixe o **U.S. Coins** em https://www.kaggle.com/datasets/sergiosaharovskiy/uscoins e coloque as imagens em
`dataset/raw/`:

```
dataset/raw/
  Jefferson Nickels, 1938-Date/
  Lincoln Cents, 1909-Date/
  Washington Quarters, 1932-1998/
```

Cada pasta é uma classe. São 3.749 imagens no total (Jefferson 1.214, Lincoln 1.285, Washington 1.250).

**Importante:** os scripts nunca alteram `dataset/raw/`. Quando um experimento precisa de outra organização das
imagens, ele cria uma cópia em outra pasta.

---

## 5. Versão original (divisão aleatória por imagem)

### 5.1 Treinar

```bash
python train.py
```

- Divisão aleatória por imagem: 80% treino / 20% validação (semente 42).
- Modelo de **4 blocos convolucionais** (`model_builder.py`).
- Adam (taxa 1e-3), lotes de 16 imagens, até 40 épocas, EarlyStopping (paciência 5).

Ao final, é criada a pasta `artifacts/`, com o modelo (`coin_classifier.keras`), os nomes das classes e os arquivos
usados na previsão.

Rode `train.py` de novo apenas se adicionar ou remover imagens ou classes, apagar `artifacts/` ou alterar
`config.py` / `model_builder.py`.

### 5.2 Testar uma imagem

```bash
python predict.py --image "dataset/raw/Jefferson Nickels, 1938-Date/Jefferson Nickel 26324_109 Reverse.jpg"
```

Mostra a porcentagem de cada classe e de `outra_moeda`. O `predict.py` usa o modelo da versão original
(`artifacts/coin_classifier.keras`).

### 5.3 Avaliar

```bash
python evaluate.py   # acurácia na validação, considerando também outra_moeda
python metrics.py    # acurácia, precisão, revocação, F1, matriz de confusão e curvas de treino
```

**Limitação:** nessa divisão, 96% das imagens de validação têm fotos da mesma série no treino. Por isso a
acurácia de validação (~95%) é otimista. A versão atual corrige isso.

---

## 6. Versão atual: divisão por série + modelo com 6 blocos

### 6.1 Por que dividir por série

As fotos do U.S. Coins vêm em **séries**: fotos de uma mesma origem, com o mesmo fundo, luz e enquadramento.
O número no nome do arquivo identifica a série (`Lincoln Cent 37633_208 Obverse.jpg` → série `37633`).
Na divisão aleatória por imagem, o modelo pode acertar reconhecendo a série, e não a moeda. Na divisão por série,
o **teste só tem séries que o modelo nunca viu**.

```
dataset/raw (3.749 fotos)
│
│  ETAPA 1 - divisão por série (make_group_split.py)
│  Séries inteiras são sorteadas (semente 42); cada série vai inteira para um lado só.
│
├── ~85% → treino_validacao (3.234 fotos)
│     │
│     │  ETAPA 2 - divisão por foto (dentro do script de treino), 80/20
│     │
│     ├── 80% → treino (2.588)
│     └── 20% → validação (646: fotos diferentes, mas das MESMAS séries do treino)
│
└── ~15% → teste (515 fotos, 7 séries nunca vistas)
      Jefferson: 30916, 26505, 212555   Lincoln: 107258, 41463   Washington: 72403, 17345
```

Regras usadas para formar as séries:

- séries que têm arquivos idênticos entre si são unidas, para que cópias nunca fiquem uma no treino e outra no teste;
- as imagens de Washington com dois números no nome (ex.: `Washington Quarter 11 6_2.jpg`) formam uma única
  coleção, com o mesmo estilo, e por isso são tratadas como uma série só.

**A validação continua tendo as mesmas séries do treino. O resultado que vale é o do TESTE.**

### 6.2 Passo 1: criar a divisão

```bash
python make_group_split.py
```

Cria `dataset/split_series_v2/` (com `treino_validacao/` e `teste/`) e o arquivo `divisao.json`.
A saída deve ser exatamente esta (se for diferente, o `dataset/raw` não é o mesmo):

```
Jefferson Nickels, 1938-Date     treino+validação:  1063 imagens (  5 grupos)   teste:  151 imagens (  3 grupos, 12.4%)
Lincoln Cents, 1909-Date         treino+validação:  1105 imagens (  5 grupos)   teste:  180 imagens (  2 grupos, 14.0%)
Washington Quarters, 1932-1998   treino+validação:  1066 imagens (  4 grupos)   teste:  184 imagens (  2 grupos, 14.7%)
```

### 6.3 Passo 2: treinar o modelo com 6 blocos

```bash
python train_group_split_profundo.py
```

Por padrão, usa `dataset/split_series_v2` e salva em `artifacts/experiments/divisao_series_v2_profundo/`.
Leva cerca de 35–45 minutos.

Opções:

| Opção | Padrão | Para que serve |
|---|---|---|
| `--pasta` | `dataset/split_series_v2` | Pasta com `treino_validacao/` e `teste/` |
| `--saida` | `artifacts/experiments/divisao_series_v2_profundo` | Onde salvar os resultados |
| `--semente` | 42 | Semente do treino (a divisão não muda) |
| `--epocas` | 40 | Número máximo de épocas |

Para repetir com outras sementes (recomendado: 42, 1 e 2):

```bash
python train_group_split_profundo.py --saida artifacts/experiments/divisao_series_v2_profundo_semente1 --semente 1
python train_group_split_profundo.py --saida artifacts/experiments/divisao_series_v2_profundo_semente2 --semente 2
```

Arquivos gerados em `--saida`: `resultados.json`, `matriz_confusao_teste.png`, `curvas_treinamento.png`,
`historico.json`, `tabela_divisao_series.tex` e o modelo `modelo_divisao_series.keras`. O script não sobrescreve
uma pasta que já tem resultado.

### 6.4 O modelo com 6 blocos

É o modelo original (`model_builder.py`) com **dois blocos a mais** (MaxPooling + Conv2D 3×3 com 256 filtros)
antes do GlobalAveragePooling. Nada mais muda (aumento de dados, dropout, otimizador, lote, épocas, EarlyStopping).
O `model_builder.py` não é alterado: o script monta o modelo original e insere os dois blocos.

| | Modelo original (4 blocos) | Modelo atual (6 blocos) |
|---|---|---|
| Blocos convolucionais | 32 → 64 → 128 → 256 | 32 → 64 → 128 → 256 → **256 → 256** |
| Mapa antes do GlobalAveragePooling | 28×28 | 7×7 |
| Área da imagem que cada unidade da última camada "vê" (campo receptivo) | 38×38 px | ~158×158 px |
| Parâmetros | 421.699 | 1.601.859 |

Com o campo receptivo maior, o modelo consegue combinar partes da moeda (texto em volta + desenho do centro),
em vez de decidir por pedaços pequenos da imagem.

### 6.5 Resultados (teste com 515 fotos de séries nunca vistas, 3 sementes)

| Modelo | Validação (séries vistas) | **Teste (séries novas)** | Jefferson | Lincoln | Washington |
|---|---|---|---|---|---|
| 4 blocos (`train_group_split.py`) | 92,9% ± 7,3 | **60,4% ± 9,1** | 66,2% | 80,9% | 35,5% |
| 6 blocos (`train_group_split_profundo.py`) | 98,3% | **70,7% ± 0,9** | 68,2% | 86,3% | 57,4% |

Nas colunas das moedas: revocação média das 3 sementes.

Para treinar o modelo de 4 blocos na mesma divisão (para comparação):

```bash
python train_group_split.py --pasta dataset/split_series_v2 --saida artifacts/experiments/divisao_series_v2
```

---

## 7. (Opcional) Treinar com o dataset LPR

O LPR é um dataset do Roboflow Universe com fotos de moedas americanas marcadas com caixas (detecção).
As moedas são recortadas e **somadas ao treino**. O teste continua sendo as mesmas 515 fotos do U.S. Coins.

**Atenção:** o LPR contém cópias de imagens do U.S. Coins e do World Coins (uCoin). O script de recorte retira as
imagens das 7 séries do teste pelo nome do arquivo.

### 7.1 Passo 1: baixar

1. Acesse https://universe.roboflow.com/lpr-2gfn3/test-krqq2
2. Escolha a **versão v4** e exporte no formato **Pascal VOC**.
3. Extraia o zip (`test.v4i.voc`) e coloque o conteúdo em `dataset/lpr_original/`.

### 7.2 Passo 2: recortar as moedas

```bash
python recortar_lpr.py
```

O script:

- usa uma cópia de cada foto original (o Roboflow gera cópias aumentadas; dá preferência à pasta `valid`);
- separa as imagens das **séries do teste** em `dataset/lpr_recortes/series_do_teste/` (não entram no treino);
- ignora as caixas que marcam só a data (`date`) ou a marca da casa da moeda (`D`, `S`, `P`, `W`);
- ignora outras moedas e anos fora das classes (dime, Buffalo, Indian, quarters de estados etc.);
- recorta cada caixa com 8% de margem e deixa o recorte quadrado;
- salva o relatório em `dataset/lpr_recortes_info.json`.

A saída esperada é:

```
Fotos originais (sem cópias aumentadas): 3687
  Jefferson Nickels, 1938-Date       934
  Lincoln Cents, 1909-Date           814
  Washington Quarters, 1932-1998     781
Recortes das séries do TESTE v2 (...): 343
```

### 7.3 Passo 3: juntar com a divisão v2

É preciso ter rodado antes o `make_group_split.py` (seção 6.2).

```bash
python montar_v2_extra.py --extra dataset/lpr_recortes --saida dataset/split_series_v2_lpr
```

Resultado: `treino_validacao` com 5.763 imagens (3.234 do U.S. Coins + 2.529 do LPR) e o mesmo `teste` (515).
A subpasta `series_do_teste` não é copiada.

### 7.4 Passo 4: treinar

Modelo com 6 blocos:

```bash
python train_group_split_profundo.py --pasta dataset/split_series_v2_lpr --saida artifacts/experiments/divisao_series_v2_lpr_profundo
```

Modelo com 4 blocos:

```bash
python train_group_split.py --pasta dataset/split_series_v2_lpr --saida artifacts/experiments/divisao_series_v2_lpr
```

---

## 8. Análises do modelo

Todos os scripts abaixo **só leem** os dados e os modelos e salvam os resultados em pastas novas dentro de
`artifacts/experiments/`.

| Script | O que faz | Exemplo |
|---|---|---|
| `oclusao_v2.py` | Cobre o centro, a borda ou o fundo da moeda e converte para tons de cinza, nas fotos do teste; compara um ou mais modelos | `python oclusao_v2.py --modelos <modelo1.keras> <modelo2.keras> --nomes base profundo` |
| `gradcam_v2.py` | Mapas de calor (Grad-CAM) de um ou mais modelos nas mesmas fotos do teste e % do mapa dentro da moeda | `python gradcam_v2.py --modelos <modelo1.keras> <modelo2.keras> --nomes base profundo` |
| `avaliar_pasta.py` | Avalia qualquer modelo numa pasta com uma subpasta por classe | `python avaliar_pasta.py --modelo <modelo.keras> --pasta dataset/controle_eua --nome profundo_controle` |
| `amostra_divisao.py` | Figuras com exemplos de cada série (treino em azul, teste em vermelho) | `python amostra_divisao.py` |

### 8.1 Variantes do modelo com 6 blocos

Cada variante muda **uma única coisa** no aumento de dados do `train_group_split_profundo.py`:

| Script | Mudança |
|---|---|
| `train_group_split_profundo_rotacao.py` | Rotação de ±18° → qualquer ângulo |
| `train_group_split_profundo_escala.py` | Zoom de ±8% → ±30% e deslocamento de ±10% |
| `train_group_split_profundo_cutout.py` | Um quadrado cinza de 56×56 px em posição aleatória nas fotos de treino (cutout) |
| `train_group_split_cinza.py` | Modelo de 4 blocos que recebe a imagem em tons de cinza |

### 8.2 Experimentos da versão original

| Script | O que faz |
|---|---|
| `controle_modelo_original.py` | Avalia o modelo original em `dataset/controle_eua` (fotos de outro site) |
| `testar_celular.py` | Avalia um modelo em `dataset/celular/teste_celular` (620 fotos de celular) |
| `checar_duplicatas.py` | Procura arquivos idênticos entre treino e validação |
| `occlusion_test.py` | Oclusão e cor na validação do modelo original |
| `treinar_aumento.py` | Variantes de aumento de dados testadas no celular |
| `degradacoes_validacao.py` / `fatores_celular.py` | Degradações (desfoque, luz, escurecimento, resolução) e fatores das fotos de celular |

---

## 9. Estrutura do projeto

| Arquivo | Função |
|---|---|
| `config.py` | Caminhos e configurações (tamanho da imagem, lote, semente, divisão) |
| `dataset_loader.py` | Carrega o dataset e faz a divisão treino/validação |
| `model_builder.py` | Arquitetura original da CNN (4 blocos, ~422 mil parâmetros) |
| `train.py` | Treino da versão original |
| `evaluate.py` / `metrics.py` | Métricas e matriz de confusão |
| `predict.py` | Classifica uma imagem (modelo da versão original) |
| `openset_utils.py` | Cálculo da probabilidade de `outra_moeda` |
| `io_utils.py` | Leitura e escrita de arquivos JSON |
| `make_group_split.py` | Cria a divisão por série (`dataset/split_series_v2`) |
| `train_group_split.py` | Treina o modelo de 4 blocos na divisão por série |
| `train_group_split_profundo.py` | Treina o modelo de 6 blocos na divisão por série |
| `recortar_lpr.py` / `montar_v2_extra.py` | Preparam o dataset LPR e o juntam ao treino |

## 10. Dados usados

| Pasta | Origem | Uso |
|---|---|---|
| `dataset/raw` | U.S. Coins – Sergio Saharovskiy, Kaggle (2023) | Treino, validação e teste |
| `dataset/lpr_original` | "test" Dataset – LPR, Roboflow Universe (2025), https://universe.roboflow.com/lpr-2gfn3/test-krqq2, versão v4, Pascal VOC, licença CC BY 4.0 | Só treino (opcional) |
| `dataset/controle_eua` | World Coins – wanderdust, Kaggle (2019), pastas de 1 e 5 centavos americanos, conferidas à mão | Só avaliação |
| `dataset/celular/teste_celular` | us-currency-recognition (UNC Charlotte), GitHub `faustogerman/us-currency-recognition`, licença Apache-2.0 | Só avaliação |

Esses dados não estão no repositório.

Citação do LPR (BibTeX):

```bibtex
@misc{test-krqq2_dataset,
  title        = {test Dataset},
  type         = {Open Source Dataset},
  author       = {LPR},
  howpublished = {\url{https://universe.roboflow.com/lpr-2gfn3/test-krqq2}},
  url          = {https://universe.roboflow.com/lpr-2gfn3/test-krqq2},
  journal      = {Roboflow Universe},
  publisher    = {Roboflow},
  year         = {2025},
  month        = {feb},
  note         = {visited on 2026-10-05}
}
```
