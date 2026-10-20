# Coin Classifier

Classificação de moedas norte-americanas com redes neurais convolucionais **treinadas do zero**, avaliada em
**séries de fotografias nunca vistas** no treinamento.

Trabalho de Conclusão de Curso (Ciência da Computação, PUCRS) de **Kamilla de Paula Borges** e **Miguel Gheno dos
Santos**, com orientação do Prof. Filipo Novo Mór.

---

## Resumo

O classificador reconhece três tipos de moeda:

| Classe | Moeda |
|---|---|
| `Jefferson Nickels, 1938-Date` | 5 centavos |
| `Lincoln Cents, 1909-Date` | 1 centavo |
| `Washington Quarters, 1932-1998` | 25 centavos |

A solução é um **ensemble de três redes de seis blocos convolucionais** (sementes 42, 1 e 2), treinadas do zero com
aumento de dados por **balanço de branco** e **cutout**, por 40 épocas. As três redes são combinadas pela **média das
probabilidades** (*soft voting*). O ensemble é usado em **duas configurações**, ambas tratadas como soluções principais:

- **Sem recorte:** a foto inteira entra nas redes.
- **Com recorte:** a moeda é localizada automaticamente (transformada de Hough), recortada e o fundo é pintado de
  branco antes de entrar nas mesmas redes.

### Por que a avaliação é por série

As fotos do conjunto U.S. Coins vêm em **séries** (mesma origem, fundo, luz e enquadramento). Numa divisão aleatória,
fotos da mesma série caem no treino e no teste, e o modelo pode acertar reconhecendo a **fotografia**, e não a moeda.
O modelo inicial acertava **95%** numa divisão aleatória, mas só **60,4%** em séries nunca vistas. Por isso, todo o
teste deste trabalho usa **séries inteiras que não aparecem no treino**.

### Resultados

| Conjunto | Média dos 3 modelos (sem recorte) | Ensemble sem recorte | Ensemble com recorte |
|---|---|---|---|
| Teste: 7 séries nunca vistas (515 imagens) | 92,6 ± 1,4% | **94,0%** | **96,1%** |
| *World Coins*: outra fonte (78 imagens) | 92,3 ± 3,4% | **97,4%** | **93,6%** |
| Avaliação final: fotos novas de moedas físicas | *(pendente)* | *(pendente)* | *(pendente)* |

- **Dependência da cor:** sem cor (tons de cinza), o modelo inicial perdia **25,2 pontos**; o ensemble sem recorte
  perde **1,6**. O modelo passou a reconhecer a moeda pelo **desenho** (sobretudo as inscrições da borda), e não pela cor.
- **Comparação foto a foto (McNemar):** no teste, o recorte foi melhor (p = 0,035, tratado como indicativo); no
  World Coins, a diferença não foi significativa (p = 0,375). Nenhuma configuração é declarada vencedora.
- **Tempo por imagem** (notebook Intel Core Ultra 7 155H, sem GPU): ~0,11–0,15 s sem recorte e ~0,22 s com recorte.

---

## Estrutura do projeto

```
Coin_Classifier/
├── config.py, dataset_loader.py, model_builder.py, metrics.py, io_utils.py   # base (usados por todos)
├── make_group_split.py                     # divisão por série
├── train_*.py                              # treinos (evolução do modelo)
├── avaliar_ensemble.py, prever_ensemble.py # solução sem recorte
├── prever_moeda_recortada.py, avaliar_ensemble_recorte.py   # solução com recorte
├── congelar_solucao.py                     # registro (hash) da solução congelada
├── avaliacao_final.py                      # avaliação final (fotos novas)
├── analisar_ensemble.py, comparar_pipelines.py, oclusao_*.py, gradcam_*.py, medir_tempo*.py, ...   # análises
├── experimentos_antigos/                   # variantes descartadas (mantidas para consulta)
├── exclusoes_extra_ucoin_v2.csv            # lista de exclusões do World Coins
├── dataset/                                # NÃO versionado (ver "Dados")
└── artifacts/                              # NÃO versionado: modelos e resultados
```

---

## Dados

As imagens **não estão no repositório**.

| Conjunto | Fonte | Como montar |
|---|---|---|
| **U.S. Coins** (3.749 imagens) | Kaggle, Sergio Saharovskiy | Baixar e colocar as três pastas de classe em `dataset/raw/` |
| **Divisão por série** | gerada | `python make_group_split.py` → `dataset/split_series_v2/` (`treino_validacao/` com 3.234 imagens e 14 séries; `teste/` com 515 imagens e 7 séries) |
| **World Coins** (78 imagens) | Kaggle, `wanderdust/coin-images` (imagens em sua maioria do catálogo uCoin) | `python montar_extra_ucoin_v2.py --origem <pasta do World Coins>` → `dataset/extra_ucoin_limpov2/` |
| **Avaliação final** | fotos tiradas pelos autores | `dataset/teste_final/` + `fotos_teste_final.csv` + `exemplares_teste_final.csv` |

As imagens do World Coins podem estar protegidas por direitos autorais e **não são redistribuídas**. O repositório
contém apenas o procedimento de montagem (`montar_extra_ucoin_v2.py` e `exclusoes_extra_ucoin_v2.csv`), que reconstrói
o mesmo conjunto a partir da fonte original.

---

## Ambiente

- Python 3.14.3, Windows 11, sem GPU.
- TensorFlow 2.22.0rc0 e Keras 3.16 (`keras-nightly`), ambos versões de pré-lançamento.
- numpy, Pillow, matplotlib e **opencv-python** (necessário para a configuração com recorte).
- Lista completa de versões: `pacotes_ambiente.txt`.

> **Importante:** nesta versão do TensorFlow, as otimizações oneDNN deixam a inferência cerca de 100 vezes mais lenta
> neste processador (as previsões não mudam). Antes de avaliar ou medir tempo, desligue no PowerShell:
> ```powershell
> $env:TF_ENABLE_ONEDNN_OPTS="0"
> ```

Nos comandos abaixo, `$M` são os três modelos da solução:
```powershell
$M = "artifacts/experiments/divisao_series_v2_profundo_balanco_branco_cutout/modelo_divisao_series.keras", "artifacts/experiments/divisao_series_v2_profundo_balanco_branco_cutout_40ep_semente1/modelo_divisao_series.keras", "artifacts/experiments/divisao_series_v2_profundo_balanco_branco_cutout_40ep_semente2/modelo_divisao_series.keras"
```

---

## Como usar

### Classificar uma foto
```powershell
python prever_ensemble.py --modelos $M --nomes s42 s1 s2 --imagem foto.jpg      # sem recorte
python prever_moeda_recortada.py --imagem foto.jpg                               # com recorte (usa os 3 modelos por padrão)
```

### Treinar um modelo da solução
```powershell
python train_group_split_profundo_balanco_branco_cutout2.py --sem_parada --semente 1 --saida artifacts/experiments/<pasta>
```
O modelo s42 foi treinado com `train_group_split_profundo_balanco_branco_cutout.py`, que é idêntico ao `..._cutout2.py`,
mas com parada antecipada. Nesse treino, a parada não interrompeu o treinamento e a melhor época foi a 40.

### Avaliar as duas configurações
```powershell
python avaliar_ensemble.py --modelos $M --nomes s42 s1 s2 --pasta dataset/split_series_v2/teste --saida artifacts/experiments/ensemble_final_teste
python avaliar_ensemble_recorte.py --modelos $M --nomes s42 s1 s2 --pasta dataset/split_series_v2/teste --saida artifacts/experiments/recorte_teste
```
Para o World Coins, troque `--pasta` por `dataset/extra_ucoin_limpov2` e use outra `--saida`.

---

## Scripts

### Solução (congelados)

| Script | O que faz |
|---|---|
| `config.py` | Caminhos e hiperparâmetros (224×224, lote 16, Adam 1e-3, semente 42) |
| `dataset_loader.py` | Leitura e preparação das imagens |
| `metrics.py`, `io_utils.py` | Métricas (acurácia, F1, matriz de confusão) e JSON |
| `make_group_split.py` | Divisão por série (séries inteiras no treino ou no teste) |
| `train_group_split_profundo_balanco_branco_cutout2.py` | Treino do modelo final (6 blocos, balanço de branco e cutout) |
| `avaliar_ensemble.py` / `prever_ensemble.py` | Ensemble sem recorte: avaliar uma pasta / classificar fotos |
| `prever_moeda_recortada.py` / `avaliar_ensemble_recorte.py` | Ensemble com recorte: classificar fotos / avaliar uma pasta |
| `avaliacao_final.py` | Avaliação final: roda as duas configurações uma única vez nas fotos novas |
| `congelar_solucao.py` | Registra os hashes da solução (`--verificar` confere se nada mudou) |

### Evolução do modelo

| Script | Modelo |
|---|---|
| `train.py` (+ `evaluate.py`, `predict.py`, `openset_utils.py`, `evaluate_openset.py`) | Primeira versão, com divisão aleatória e reconhecimento *open-set* |
| `train_group_split.py`, `train_group_split2.py` | Modelo inicial de 4 blocos, com divisão por série |
| `train_group_split_profundo.py`, `train_group_split_profundo2.py` | 6 blocos |
| `train_group_split_profundo_balanco_branco_cutout.py` | 6 blocos + balanço de branco + cutout, com parada antecipada |
| `model_builder.py` | Construção das redes |

### Análises

| Script | Análise |
|---|---|
| `verificar_duplicatas.py`, `refinar_duplicatas.py` | Cópias exatas e quase iguais entre treino e teste |
| `conferir_pasta_externa.py` | Cópias entre o World Coins e o U.S. Coins |
| `medir_series.py`, `analisar_erros_series.py` | Brilho e tom por série; erros por série (motivaram o balanço de branco) |
| `figura_series.py`, `figura_treino_teste.py`, `folhas_por_serie.py` | Figuras e inspeção visual das séries |
| `analisar_ensemble.py` | Concordância entre os modelos e o ensemble |
| `comparar_pipelines.py` | Comparação foto a foto sem × com recorte (McNemar) |
| `occlusion_test.py`, `oclusao_v2.py`, `oclusao_ensemble.py` | Teste de oclusão e de remoção da cor |
| `gradcam_v2.py`, `gradcam_v3.py`, `gradcam_imagem.py` | Mapas Grad-CAM |
| `listar_erros_pasta.py` | Lista os erros de um modelo numa pasta |
| `medir_tempo.py`, `medir_tempo_recorte.py`, `diagnostico_tempo.py` | Tempo por imagem e diagnóstico do oneDNN |
| `figura_exemplos.py` | Figura com fotos originais e recortadas (e respostas das duas configurações) |
| `avaliar_pasta.py`, `prever_imagem.py`, `controle_modelo_original.py` | Avaliação ou previsão com um único modelo |

### `experimentos_antigos/`

Variantes testadas e descartadas: tons de cinza, cutout sozinho, balanço de branco sozinho, escala, rotação, dourado,
fotos de celular e bases LPR/Roboflow. Ficam no repositório para consulta. Para rodar um deles:
```powershell
$env:PYTHONPATH = "."
python experimentos_antigos/<script>.py
```

---

## Solução congelada

Antes da avaliação final, os três modelos e todos os scripts da solução foram **congelados**, com o registro do seu
SHA-256 em `PROTOCOLO_CONGELADO.md`. O mesmo arquivo descreve as regras definidas
antes da coleta das fotos novas: execução única, nenhuma foto descartada e nenhuma configuração declarada vencedora.

Para conferir se nada foi alterado:
```powershell
python congelar_solucao.py --verificar
Get-FileHash prever_moeda_recortada.py, avaliar_ensemble_recorte.py, avaliacao_final.py -Algorithm SHA256
```

---

## Limitações

- Reconhece apenas as três classes; diante de uma moeda de outro tipo, responde sempre uma delas.
- Desenhos raros no treino concentram parte dos erros: Jefferson a partir de 2004/2005 (~4% da classe) e Washington
  do bicentenário (~2%).
- O conjunto de teste orientou o desenvolvimento; por isso, os resultados são confirmados no World Coins e na
  avaliação final.
