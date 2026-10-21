# Solução final congelada (2026-10-12 15:16)

- **Solução:** Ensemble de 3 CNNs (mesma arquitetura, sementes 42, 1 e 2) por soft voting
- **Regra do ensemble:** média aritmética simples das probabilidades das 3 redes, sem pesos; classe = maior média
- **Ordem das classes:** (1) Jefferson Nickels, 1938-Date; (2) Lincoln Cents, 1909-Date; (3) Washington Quarters, 1932-1998
- **Pré-processamento:** imagem RGB redimensionada para 224x224 (bilinear, sem manter a proporção), valores 0-255 em float32 (dataset_loader.build_single_image_tensor)
- **Pesos salvos:** estado da última época (época 40), sem checkpoint; ver a observação de cada modelo
- **Métricas:** acurácia, F1 macro, matriz de confusão, tempo por imagem
- **Ambiente:** Python 3.14.3, TensorFlow 2.22.0-rc0, Windows 11

## Modelos

| Modelo | Semente | Épocas | Teste v2 | SHA-256 |
|---|---|---|---|---|
| s42 | 42 | 40 | 92.0% | `8bde3bb84849535c124b9800208398cc9f59eabb73b2e545592db7064321d332` |
| s1 | 1 | 40 | 91.5% | `4585cd25cb26d83f893ed6bc0881a5f1146c638bc34b9c74038dfa84f4cbc0b5` |
| s2 | 2 | 40 | 94.2% | `d8bc442df93453ddef6404d6a3359cfee1b1d0303e8ae31a0f17870d93bf6a2d` |

- **s42:** treinado com EarlyStopping (paciência 5, restore_best_weights), que não interrompeu o treino; a melhor época foi a 40, então os pesos salvos são os da última época, como nos demais
- **s1:** treinado com --sem_parada (40 épocas fixas, pesos da última época)
- **s2:** treinado com --sem_parada (40 épocas fixas, pesos da última época)

## Scripts

| Arquivo | SHA-256 |
|---|---|
| config.py | `f3daa251688d033c5158debe2872b39a553cac1e8884309488fef73a414c1e26` |
| dataset_loader.py | `c69324c2519e760c9b0cff25ec2344d417112a286811ff47ae2241386af0cfd5` |
| metrics.py | `fba2d89d2a1ef94dbc7c77981dbfd66090caefd735a9d75c1a5738a4d9eb0395` |
| io_utils.py | `895c7c22df271d45437d2c4849857205eb8daffc4c7eea06ac3d210e810f2989` |
| make_group_split.py | `35a7b9d9bac1e2df4c56933fd251626f6cc4dcd7673a3cea0285204efc5218f2` |
| train_group_split_profundo_balanco_branco_cutout2.py | `c39d0f9f67de43b55e75c2c7b34dfb329da45fa6b5378d6691d660360c0bb6ce` |
| avaliar_ensemble.py | `7010472418e61737503e8e25942c8abd40a504bee955af720a7363bb0f208478` |
| prever_ensemble.py | `5ec0301b7abdb273750b249d5bf7853afc756e08bcc148ce0ba3a1c077d4e2c0` |

## Informações complementares (acrescentadas manualmente)

### Script de treino de cada modelo
- **s42:** `train_group_split_profundo_balanco_branco_cutout.py`, SHA-256 `2e5e5f2a99a5bc1ea4e04325ba085be67cd74ca3142fd76dd8c175594c6d042f`. É idêntico ao `..._cutout2.py`, exceto pela opção `--sem_parada` (no script antigo, a parada antecipada está sempre ligada).
- **s1 e s2:** `train_group_split_profundo_balanco_branco_cutout2.py --sem_parada` (hash na tabela de scripts acima).

### Dados
- Dataset: U.S. Coins (Kaggle, Sergio Saharovskiy), pasta `dataset/raw`, 3.749 imagens (Jefferson 1.214; Lincoln 1.285; Washington 1.250). Não foi alterado.
- Divisão por série (`make_group_split.py` → `dataset/split_series_v2`):
  - treino+validação: 3.234 imagens (14 séries); o script de treino separa 80/20 → 2.588 treino / 646 validação, sempre com a semente fixa 42;
  - teste: 515 imagens de 7 séries nunca vistas no treino (Jefferson 30916, 26505, 212555; Lincoln 107258, 41463; Washington 72403, 17345). Por classe: Jefferson 151, Lincoln 180, Washington 184.
- Séries com arquivos idênticos (mesmo MD5) foram agrupadas, para que nenhuma cópia exata fique em treino e teste ao mesmo tempo.
- Conjunto externo World Coins (`dataset/extra_ucoin_limpov2`): 78 imagens (Jefferson 28; Lincoln 20; Washington 30), montado com `montar_extra_ucoin_v2.py` e `exclusoes_extra_ucoin_v2.csv`.

### Treinamento (igual nos 3 modelos)
- CNN de 6 blocos convolucionais, treinada do zero (sem pesos pré-treinados), 1.601.859 parâmetros.
- Entrada 224×224; Adam, taxa de aprendizado 0,001; lote de 16; entropia cruzada categórica; 40 épocas.
- Aumentos de dados (só no treino): espelhamento horizontal, rotação (±0,05 volta), zoom (±8%), contraste (±8%), balanço de branco (probabilidade 0,8; 2.850–7.500 K; exposição 0,6–1,3) e cutout (quadrado cinza de 56×56 px).
- O que a semente muda: só a aleatoriedade do treino (pesos iniciais, ordem dos lotes e aumentos de dados). As partições são as mesmas nos três modelos.

### Papel de cada conjunto
- **Teste v2 (515 imagens):** avaliação exploratória/de desenvolvimento. Orientou as escolhas do modelo (arquitetura, aumentos de dados, número de épocas), porque a validação (mesmas séries do treino) ficou entre 98,9% e 100% e não diferenciava as alternativas.
- **World Coins (78 imagens):** avaliação externa, sem uso para treinar ou escolher o modelo.
- **Avaliação final:** fotos novas, coletadas depois deste congelamento, com protocolo definido antes de ver as previsões. **(pendente)**

### Resultados registrados no momento do congelamento

| Conjunto | s42 | s1 | s2 | Média ± DP | Ensemble |
|---|---|---|---|---|---|
| Teste v2 (515) – acurácia | 92,0% | 91,5% | 94,2% | 92,6 ± 1,4% | 94,0% |
| Teste v2 – F1 macro | 0,919 | 0,914 | 0,940 | – | 0,939 |
| World Coins (78) – acurácia | 94,9% | 93,6% | 88,5% | 92,3 ± 3,4% | 97,4% |
| World Coins – F1 macro | 0,945 | 0,933 | 0,878 | – | 0,972 |

O desvio padrão vale só para os modelos individuais. O ensemble é um classificador único, sem desvio.

### Ambiente
- Computador: notebook Intel Core Ultra 7 155H (16 núcleos, 22 lógicos), 32 GB de RAM, sem GPU.
- Python 3.14.3, Windows 11.
- TensorFlow 2.22.0rc0 e Keras 3.16.0.dev2026092804 (keras-nightly): ambas são versões de pré-lançamento.
- Outros pacotes principais: numpy 2.4.3, pillow 12.1.1, matplotlib 3.10.8.
- Lista completa de pacotes: `pacotes_ambiente.txt` (nesta pasta).
- O Python, o TensorFlow e o Keras não serão atualizados até o fim da avaliação final, porque versões diferentes podem alterar levemente os resultados.
- Execução com `TF_ENABLE_ONEDNN_OPTS=0` (otimizações oneDNN desligadas) para medir o tempo e na avaliação final. Com o oneDNN ligado (padrão), esta versão de pré-lançamento do TensorFlow ficou cerca de 100 vezes mais lenta neste processador (~3,9 s por imagem). As previsões foram idênticas com e sem oneDNN, tanto no ensemble sem recorte quanto no ensemble com recorte (conferido no teste v2: mesmas acurácias e mesma matriz de confusão; saídas em `ensemble_final_teste_sem_onednn` e `recorte_teste_sem_onednn`).

### Regras a partir deste congelamento
- Os 3 modelos e os scripts listados não serão alterados.
- Antes da avaliação final, rodar `python congelar_solucao.py --verificar`. O resultado tem que ser "OK".
- Nenhuma foto da avaliação final será descartada por causa da previsão do modelo.

## Análises feitas depois do congelamento (sem alterar modelos nem scripts congelados)

### Concordância entre os modelos (`analisar_ensemble.py`)

| | Teste v2 (515) | World Coins (78) |
|---|---|---|
| Os 3 modelos acertam | 449 (87,2%) | 65 (83,3%) |
| Os modelos discordam | 57 (11,1%) | 12 (15,4%) |
| Ensemble acerta, com pelo menos 1 modelo errado | 35 (6,8%) | 11 (14,1%) |
| Ensemble erra, com pelo menos 1 modelo certo | 15 (2,9%) | 0 |
| Os 3 modelos erram | 16 (3,1%) | 2 (2,6%) |

Teste v2, por modelo (erros → corrigidos pelo ensemble / novos): s42 41 → 13 / 3; s1 44 → 16 / 3; s2 30 → 10 / 11.

### Fotos duplicadas (`verificar_duplicatas.py` + `refinar_duplicatas.py`)
- Treino+validação × teste: 0 cópias exatas (MD5); nenhum par quase igual (correlação máxima de pixels 0,92; cópias reais dão 1,00 e pares de moedas diferentes chegam a 0,91).
- Dentro do teste: 4 grupos de arquivos idênticos (Jefferson 30916_155 = _24 = _34 = _56; Jefferson 30916_43 = _96; Lincoln 41463_19 = _64; Lincoln 41463_3 = _70), 6 cópias a mais.
- Sensibilidade (509 imagens, 1 por grupo): média dos modelos 93,1 ± 1,4%; ensemble 94,5%. O resultado oficial continua sendo o das 515 imagens.
- Limitação: o método não detecta a mesma moeda fotografada novamente.

### Tempo por imagem (`medir_tempo.py`; uma imagem por vez, 100 imagens do teste após 10 de aquecimento; leitura + pré-processamento + rede)

| Configuração | Individual (média / mediana / p95) | Ensemble (média / mediana / p95) |
|---|---|---|
| oneDNN desligado (`TF_ENABLE_ONEDNN_OPTS=0`) | 38,9 / 37,2 / 55,8 ms | 109,9 / 104,8 / 153,6 ms |
| oneDNN ligado (padrão) | 3.885,8 / 3.820,4 / 4.314,7 ms | 11.636,8 / 11.369,8 / 12.930,1 ms |

O ensemble leva cerca de 2,8–3,0 vezes o tempo do modelo individual nas duas configurações.

### Oclusão com o ensemble (`oclusao_ensemble.py`)

Teste v2 (515 imagens), mesmas máscaras do `occlusion_test.py`:

| Condição | Ensemble | Queda | Média dos 3 modelos |
|---|---|---|---|
| Original | 94,0% | – | 92,6% |
| Centro coberto | 86,0% | −8,0 | 83,0% |
| Borda coberta | 56,5% | −37,5 | 52,9% |
| Fundo coberto | 84,1% | −9,9 | 82,7% |
| Tons de cinza | 92,4% | −1,6 | 87,4% |

- Classe mais afetada em cada condição: centro coberto → Jefferson (59,6%); borda coberta → Washington (19,0%); cinza → Lincoln (87,2%).
- Para comparação, o modelo inicial caía −8,5 (centro), −11,7 (borda), −6,8 (fundo) e −25,2 (cinza): o modelo final depende muito menos da cor e mais da borda da moeda.

## Configuração B: ensemble com recorte da moeda (congelada em 2026-10-14)

- **O que é:** os mesmos 3 modelos da configuração A e a mesma regra do ensemble (média simples das probabilidades). Muda só o pré-processamento da foto antes da rede.
- **Pré-processamento (`prever_moeda_recortada.isolate_coin`):**
  1. aplica a rotação gravada pelo celular (EXIF);
  2. procura o círculo da moeda (HoughCircles numa cópia de 400 px; raio entre 20% e 55% do lado; param1 = 100; param2 = 30);
  3. recorta um quadrado em volta do círculo, com margem de 8%;
  4. pinta de branco (255) o que fica fora do círculo;
  5. se nenhum círculo for encontrado, usa o quadrado central da foto, sem pintar o fundo.

  Depois disso, a imagem é redimensionada para 224×224, como na solução principal.
- **É automático:** nenhuma etapa é manual.
- **Origem:** esta configuração foi proposta depois do congelamento da configuração A (ensemble sem recorte) e avaliada nos mesmos conjuntos (teste v2 e World Coins), que já tinham sido usados no desenvolvimento. Com a concordância do orientador, as duas configurações passam a ser tratadas como soluções principais do trabalho. Essa ordem dos acontecimentos será declarada no artigo.
- **Ambiente:** o mesmo da solução principal, mais OpenCV 5.0.0.
- **Repositório:** os dois scripts estão no commit `b503d4d` (GitHub, branch `main`), com os mesmos hashes da tabela abaixo.

### Scripts

| Arquivo | SHA-256 |
|---|---|
| prever_moeda_recortada.py | `7241a53120e3a6e5ac59d71f92de63010c2d8f0a58db2c92b8367d30407be794` |
| avaliar_ensemble_recorte.py | `b224c675aa880348049fd7c7bc1ad26490ae3b0179adf5ea36786df421ab7a39` |

### Resultados registrados no momento do congelamento da variante

| Conjunto | s42 | s1 | s2 | Média ± DP | Ensemble |
|---|---|---|---|---|---|
| Teste v2 (515) – acurácia | 95,7% | 94,4% | 92,4% | 94,2 ± 1,7% | 96,1% |
| Teste v2 – F1 macro | 0,958 | 0,944 | 0,925 | – | 0,961 |
| World Coins (78) – acurácia | 94,9% | 96,2% | 89,7% | 93,6 ± 3,4% | 93,6% |
| World Coins – F1 macro | 0,944 | 0,957 | 0,885 | – | 0,929 |

Matriz do ensemble com recorte (linhas = moeda certa; colunas = resposta, na ordem Jefferson, Lincoln, Washington):

| | Teste v2 | World Coins |
|---|---|---|
| Jefferson | 141 / 4 / 6 | 27 / 1 / 0 |
| Lincoln | 0 / 179 / 1 | 2 / 17 / 1 |
| Washington | 1 / 8 / 175 | 0 / 1 / 29 |

- O círculo foi encontrado em 515 de 515 fotos do teste e em 78 de 78 do World Coins (o quadrado central não foi usado em nenhuma).
- Saídas: `artifacts/experiments/recorte_teste` e `artifacts/experiments/recorte_ucoinv2` (JSON, CSV por foto e as imagens recortadas que a rede recebeu).
- **Comparação com o pipeline original:** no teste v2 o recorte melhorou (ensemble 94,0 → 96,1%; média 92,6 → 94,2%). No World Coins, a média ficou parecida (92,3 → 93,6%) e o ensemble piorou (97,4 → 93,6%; 76 → 73 acertos de 78). Por isso, nenhum dos dois pipelines é declarado melhor antes da avaliação final.

### Regras
- `prever_moeda_recortada.py` e `avaliar_ensemble_recorte.py` não serão alterados daqui em diante.
- O OpenCV não será atualizado até o fim da avaliação final.
- Antes da avaliação final, conferir os dois hashes com `Get-FileHash prever_moeda_recortada.py, avaliar_ensemble_recorte.py -Algorithm SHA256` (o `congelar_solucao.py --verificar` só confere a solução principal).
- Na avaliação final, os dois pipelines rodam uma única vez, sobre as mesmas fotos, com `TF_ENABLE_ONEDNN_OPTS=0`.

## Duas configurações principais e regra para a avaliação final (definida em 2026-10-14, antes da coleta das fotos novas)

| | Configuração A | Configuração B |
|---|---|---|
| Nome no artigo | Ensemble | Ensemble + recorte |
| Modelos | s42, s1, s2 (os mesmos) | s42, s1, s2 (os mesmos) |
| Regra de combinação | média simples das probabilidades | média simples das probabilidades |
| Pré-processamento | foto inteira redimensionada para 224×224 | moeda isolada (`prever_moeda_recortada.isolate_coin`) e redimensionada para 224×224 |
| Script da avaliação | `avaliar_ensemble.py` | `avaliar_ensemble_recorte.py` |

- Os modelos foram treinados com fotos inteiras. O recorte é aplicado só na previsão, sem novo treino.

### Como a avaliação final será feita
1. Conferir os hashes: `python congelar_solucao.py --verificar` (configuração A) e `Get-FileHash` dos dois scripts da configuração B. Tudo tem que bater.
2. Desligar o oneDNN (`TF_ENABLE_ONEDNN_OPTS=0`).
3. Rodar as duas configurações **uma única vez**, sobre **as mesmas fotos**, sem descartar nenhuma foto.
4. Se o círculo não for encontrado em alguma foto, a configuração B usa o quadrado central, como já está no script. A foto não é descartada, e o número de fotos nessa situação é informado.

### Como os resultados serão apresentados
- **Medidas principais (para cada configuração):** acurácia e F1 macro do ensemble, matriz de confusão e acurácia por moeda.
- **Medidas complementares:** acurácia de cada modelo (média ± desvio padrão das 3 sementes) e tempo por imagem.
- **Comparação pareada:** número de fotos em que só A acertou, só B acertou, as duas acertaram e as duas erraram.
- **Não será escolhida uma configuração vencedora.** As duas serão apresentadas lado a lado. A discussão vai dizer em quais situações cada uma se saiu melhor (por exemplo: fundo, iluminação, tipo de moeda).
- Nenhum ajuste será feito depois de ver os resultados da avaliação final.

## Script da avaliação final (congelado em 2026-10-15, antes da entrega das fotos novas)

| Arquivo | SHA-256 |
|---|---|
| avaliacao_final.py | `328d8d0dd856b2d732f803522c29043246da339677a5f695ee7ed0c5f5c3ee57` |

- **O que faz:** roda as duas configurações (A, sem recorte, e B, com recorte) **uma única vez**, sobre as mesmas fotos, com os mesmos 3 modelos e a mesma regra do ensemble.
  - A usa o mesmo pré-processamento do `avaliar_ensemble.py`.
  - B usa o mesmo do `avaliar_ensemble_recorte.py`, ou seja, `prever_moeda_recortada.isolate_coin`.
- **Conferência antes de classificar:** o script confere se as pastas e as planilhas (`fotos_teste_final.csv` e `exemplares_teste_final.csv`) batem: toda foto tem uma linha, toda linha tem uma foto, e a classe da pasta é igual à da planilha. Se algo não bater, ele para **sem classificar nenhuma foto**. O modo `--conferir` faz só essa conferência.
- **Execução única:** o script se recusa a rodar se a pasta de saída já existir, para não sobrescrever uma execução anterior.
- **O que é relatado, para cada configuração:**
  - acurácia e F1 macro do ensemble e de cada modelo, com a média ± desvio padrão dos modelos;
  - matriz de confusão;
  - resultados por moeda (exemplar), por sessão, por face e por acabamento (normal ou proof);
  - média da acurácia por moeda e número de moedas com todas as fotos certas;
  - número de fotos em que o círculo da moeda não foi encontrado.
- **Comparação A × B:** contagem de fotos em que as duas acertam, só A acerta, só B acerta e as duas erram, com o **teste exato de McNemar** (só A × só B). Por causa da regra definida antes, nenhuma configuração será declarada vencedora, qualquer que seja o resultado do teste.
- **Registro:** o próprio resultado grava o SHA-256 dos 3 modelos e dos scripts usados.

### Ordem dos comandos no dia da avaliação final
1. `python congelar_solucao.py --verificar` → tem que dar "OK".
2. `Get-FileHash prever_moeda_recortada.py, avaliar_ensemble_recorte.py, avaliacao_final.py -Algorithm SHA256` → os três hashes têm que ser iguais aos deste protocolo.
3. `$env:TF_ENABLE_ONEDNN_OPTS="0"`
4. `python avaliacao_final.py ... --conferir` → tem que dizer "Pastas e planilhas conferem". Se não disser, corrige-se **só a planilha ou o nome das pastas**, nunca as fotos.
5. `python avaliacao_final.py ...` sem `--conferir` → **uma única vez**, com saída em `artifacts/experiments/avaliacao_final`.
## Acréscimo (2026-10-21): model_builder.py

- O script de treino final (`train_group_split_profundo_balanco_branco_cutout2.py`) importa `build_model` do
  `model_builder.py`, que define os 4 primeiros blocos convolucionais, o aumento de dados básico (espelhamento,
  rotação, zoom e contraste) e a parte final da rede. Esse arquivo não constava da lista de scripts congelados.
- Ele não é usado na avaliação: os scripts de avaliação carregam os arquivos `.keras`, que já guardam a arquitetura
  completa e os pesos. Só é necessário para treinar os modelos de novo.
- Última alteração no repositório: commit `acf6634` (2026-09-26), anterior ao congelamento.
- Não será alterado até o fim do trabalho.

| Arquivo | SHA-256 |
|---|---|
| model_builder.py | `e6efb83acffe25d1b54751b2b3f3032fdbbb88649e26a513bd801ceff35f9592` |