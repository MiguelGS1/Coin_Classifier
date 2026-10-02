# CLAUDE.md – TCC "Atalhos de aprendizado e generalização fora do domínio" (PUCRS)

> Documento de contexto escrito por um Claude para outros Claudes. Leia inteiro antes de responder.
> Ele resume meses de conversa: decisões, números reais, erros já cometidos e o que falta fazer.
> **Todos os números aqui vieram de saídas reais dos scripts.** Não invente números novos; se precisar
> de um valor que não está aqui, peça a saída do terminal ou o arquivo de resultado.

---

## 1. Quem são e como trabalhar com eles

- TCC II de Ciência da Computação na **PUCRS** (Escola Politécnica), em dupla:
  **Kamilla de Paula Borges** e **Miguel Gheno dos Santos**. Orientador: **Prof. Filipo Novo Mór**
  (responde pouco; se necessário, sugerir procurar a coordenação do TCC).
- O artigo é em **LaTeX, template ACM (ACM Transactions on Graphics)**, limite de **16 páginas**
  (última versão vista tinha 9 páginas).
- **Às vezes quem escreve é a Kamilla, às vezes o Miguel.** Eles costumam começar com "sou a Kamilla" ou
  "aqui é o Miguel". Se não ficar claro e isso importar (comandos, arquivos, divisão de tarefas),
  **pergunte**. Os datasets e modelos completos estão **só no computador da Kamilla**; o Miguel tem
  apenas o que ela mandar em zip.
- Responder **sempre em português**, simples e direto, com passo a passo. Eles não são especialistas em
  Git/PowerShell: dar um comando por vez e dizer o que cada um faz e o que deve aparecer.
- **Seja honesto, não agrade.** Eles pediram explicitamente avaliação sincera (inclusive se o trabalho é
  "digno de TCC"). Aponte fraquezas que a banca vai ver.
- Em algum momento a Kamilla disse "não ligo para honestidade, quero resultados bons". A postura adotada
  foi: ajudar a obter bons resultados **só com métodos legítimos** (nada de escolher semente/teste a dedo,
  nada de esconder resultado que muda a interpretação, sempre declarar o método). Mantenha isso.
- Textos para o artigo: registro de TCC, frases curtas, **sem cara de texto de IA**, fiéis ao código e
  aos números. Use `\textit{}` nos nomes das classes em inglês.

### Regras técnicas combinadas
- **Nunca alterar arquivos que já existem no repositório** (`config.py`, `train.py`, `model_builder.py`,
  `dataset_loader.py`, `openset_utils.py`, `predict.py`, `evaluate.py`, `io_utils.py`) e nunca mexer em
  `dataset/raw`. Tudo novo vai em **scripts novos**, testados antes, entregues como arquivo separado.
- Scripts novos só **leem** o que existe e salvam em **pastas novas** (`artifacts/experiments/<nome>/`).
  Nunca sobrescrever `artifacts/coin_classifier.keras` nem resultados antigos; se o destino já existir,
  o script deve **parar e avisar**.
- Ao entregar um script, informar o **número de linhas**, para ela conferir com
  `(Get-Content arquivo.py).Count` (já houve arquivo truncado ao colar; por isso mandar como anexo).
- Ambiente da Kamilla: **Windows + PowerShell no VS Code**, pasta
  `C:\Users\kamil\Downloads\Coin_Classifier_git`. Usar o **Python do sistema** (3.14, com TensorFlow e
  Keras 3.16). O `.venv` está **vazio**: se aparecer `(.venv)` no prompt, mandar rodar `deactivate`.
  OpenCV instalado (`opencv-python`). Arquivos recebidos chegam em `$HOME\Downloads` e são copiados com
  `Copy-Item "$HOME\Downloads\arquivo.py" .`.
- Git: ela já errou o fluxo (commit sem `git add`). Sempre: `git status` → `git add <arquivos>` →
  `git status` → `git commit -m "..."` → `git push origin main`. Aviso de "LF will be replaced by CRLF" é normal.
- `.gitignore` contém `dataset/` e `artifacts/` (dados e modelos **não vão para o GitHub**), além de
  scripts ignorados de propósito (ver seção 9).

---

## 2. O tema e o escopo atual (decidido)

**Título:** *Atalhos de aprendizado e generalização fora do domínio em redes convolucionais: um estudo de
caso com moedas.* (em inglês, sugestão: *Shortcut Learning and Out-of-Domain Generalization in
Convolutional Neural Networks: A Case Study with Coins*)

As moedas são o **estudo de caso**; o tema é o comportamento de uma CNN fora do domínio de treino. O
trabalho **não é inédito no sentido amplo** (atalhos e queda fora do domínio são conhecidos); a
contribuição é medir e analisar com cuidado num caso concreto, com métodos já estabelecidos. Isso é
adequado para TCC de graduação.

| QP | Pergunta | Papel na história | Situação |
|---|---|---|---|
| QP1 | A validação comum superestima o desempenho fora do domínio? Quanto? | Mostrar o problema | **Feita** |
| QP2 | Em que o modelo se apoia para decidir? Ele aprende atalhos? | Explicar | **Feita** |
| QP3 | Que degradação mais prejudica o modelo: mudança de cor ou perda de detalhe? | Testar a explicação | **A fazer (Miguel)**; scripts prontos |
| QP4 | O aumento de dados reduz a perda no celular? | Tentar corrigir | **Feita**; falta repetir com sementes |

**História:** o modelo parece ótimo na validação (95,1%) → mas cai para 90% em fotos boas de outra fonte
e para 51% (balanceada) em fotos de celular (QP1) → porque se apoia em atalhos: cor, fundo, fonte da foto
(QP2) → degradações controladas mostram o que mais o derruba (QP3) → o aumento de dados não corrige, só
troca um atalho por outro (QP4). Conclusão geral: acurácia de validação não basta; é preciso testar fora
do domínio, olhar acurácia balanceada e matriz de confusão.

**Decisões já tomadas (não reabrir sem motivo):**
- QP1 **sem** divisão por série do U.S. Coins (a Kamilla não quis). Consequência: não afirmar que o
  vazamento *causa* a queda 95→51; dizer que a validação é otimista e que o vazamento *contribui*.
- QP3 entra como **diagnóstico do modelo com protocolo já estabelecido**, citando Dodge e Karam (2016),
  Hendrycks e Dietterich (2019) e Geirhos et al. (2019/2020). O Miguel questionou se era necessária, já
  que isso foi estudado; a resposta foi: não é obrigatória, não é descoberta nova, mas testa o atalho de
  cor da QP2 por outro caminho e liga QP2 à QP4. Se faltar espaço/tempo: só **luz amarelada e desfoque**.
- **Prioridade de reforço:** repetir `atual`, `sem_aumento` e `forte` com **mais 2 sementes cada** (≈6
  treinos de ~35 min). É o ponto mais fraco do trabalho (uma execução por variante). Se só der uma coisa,
  fazer as sementes antes da QP3.
- **Sai do artigo:** rejeição de moedas desconhecidas por protótipos (antigo Experimento B/C) – no
  máximo um parágrafo de trabalho futuro; Grad-CAM; experimento antigo de aumento (`train_augmentation.py`).
- **Volta ao artigo (na QP2):** a classe `outra_moeda` (antigo "Experimento E") como evidência de
  **atalho de fonte**, não como sistema de rejeição.
- Possíveis trabalhos futuros já discutidos: fotos de celular no treino (dividindo o `teste_celular` por
  sessão de fotos), rede pré-treinada (MobileNetV2) como comparação, test-time augmentation, divisão por série.

---

## 3. O código original do projeto (repositório MiguelGS1/Coin_Classifier)

Arquivos originais: `config.py`, `dataset_loader.py`, `model_builder.py`, `train.py`, `evaluate.py`,
`predict.py`, `openset_utils.py`, `io_utils.py`, `README.md`.

- **Modelo** (`model_builder.py`): entrada 224×224×3 → `Rescaling(1/255)` → aumento de dados
  (`aug_flip` espelhamento horizontal, `aug_rotation` 0,05 ≈ ±18°, `aug_zoom` 0,08, `aug_contrast` 0,08)
  → 4 blocos Conv2D 3×3 ReLU `same` com 32, 64, 128, 256 filtros, MaxPooling 2×2 após os 3 primeiros
  → GlobalAveragePooling → Dropout 0,35 → Dense 128 ReLU com L2 1e-4 (camada `embedding`) → Dense
  softmax (`class_probs`). **421.699 parâmetros** (~422 mil). Campo receptivo de cada unidade da 4ª
  camada: 38×38 pixels (não "vê" a moeda inteira; o GAP junta tudo).
- **Treino** (`train.py`): Adam lr 1e-3, entropia cruzada categórica, lote 16, **até 40 épocas** (o
  `EPOCHS = 18` do `config.py` NÃO é usado; o 40 está no `train.py`), EarlyStopping em `val_loss` com
  paciência 5 e `restore_best_weights`. Modelo final: **melhor época 38 de 40**. Com 18 épocas, a
  validação era 93,3%. Cada treino leva ~30–35 min no computador da Kamilla.
- **Divisão**: `image_dataset_from_directory` com `validation_split=0.2`, `seed=42`, por imagem, sem
  conjunto de teste → 3.000 treino / 749 validação. Classes em ordem alfabética: Jefferson, Lincoln,
  Washington.
- **Rejeição open-set** (`openset_utils.py`): protótipo = média dos embeddings de treino de cada classe;
  τc = percentil 10 da confiança da classe correta na validação = **0,8936**; τd = percentil 90 da
  distância ao próprio protótipo = **5,1368**; sinais s_c e s_d; u = min(0,95; 0,6·s_c + 0,4·s_d) vira a
  probabilidade de `outra_moeda`, e as conhecidas são multiplicadas por (1−u). Prova de que quase nunca
  vence: se c ≥ τc, s_c = 0, u < 0,4 e a classe prevista fica com ≥ 0,894·0,6 ≈ 0,54.

---

## 4. Dados

| Pasta | Origem | Conteúdo | Uso |
|---|---|---|---|
| `dataset/raw` | **U.S. Coins** (Sergio Saharovskiy, Kaggle 2023) | 3.749 imagens: Jefferson 1.214, Lincoln 1.285, Washington 1.250. Fotos de moedas em holders de certificação, estilo catálogo/estúdio. Frente e verso na mesma classe | Treino/validação |
| `dataset/controle_eua` | **World Coins** (wanderdust, Kaggle 2019), pastas `train/206` (1 Cent) e `train/207` (5 Cents) | 26 Lincoln + 24 Jefferson, **conferidos à mão** (removidos: Indian Head ×2 do 1 cent; Liberty "V", Shield e Buffalo ×2 do 5 cents). Fotos do site uCoin: fundo liso, marca d'água lateral. Sem Washington | QP1 (fotos boas de outra fonte) e QP2 |
| `dataset/celular/teste_celular` | **us-currency-recognition** (Gambill, German Jimenez, Wilk, Wygal, Ademiluyi – UNC Charlotte, GitHub `faustogerman/us-currency-recognition`, Apache-2.0) | **620 recortes**: Lincoln 313, Jefferson 266, Washington 41. Fotos de celular (4032×3024), várias moedas por foto; recortadas pelas caixas Pascal VOC, reduzidas a ≤512 px e **conferidas uma a uma** | QP1, QP3, QP4. **Só teste** |
| `dataset/celular/teste_celular_quarters_1999_2021` | idem | 77 quarters com a frente de 1999–2021 (texto em outra posição) | **Não usado** (opcional) |
| `dataset/unknown` | World Coins | "Próximas": Canadá 1c (pasta 22), 5c (23), 25c (25) – 52 cada; Japão 100 ienes (97) – 28. Total 184 | Antigo experimento de rejeição |
| `dataset/unknown_distantes` | World Coins | "Distantes": RU 20p (18), 50p (19), 2 libras (21) – 52 cada; Dinamarca 1 coroa (49) – 28. Total 184 | Antigo experimento de rejeição |
| `dataset/unknown_excluidas` | World Coins | Austrália 50c (6) e Japão 5 ienes (94), tiradas da análise | Não usado |
| `testeExperimentoE/` | World Coins + OpenCV | Scripts e dados da classe `outra_moeda` (ver QP2) | Guardado fora do fluxo |

**Nomes das classes = nomes das pastas:** `Jefferson Nickels, 1938-Date`, `Lincoln Cents, 1909-Date`,
`Washington Quarters, 1932-1998`. Washington vale só até 1998: frente com "LIBERTY" no alto (o desenho de
2021 também tem esse layout) ou verso com a **águia**. Quarters de estados, parques e o verso de 2021
("Crossing the Delaware") **não** pertencem à classe.

**Sobre o `teste_celular`:** dos 946 recortes de moedas (376 penny, 284 nickel, 286 quarter), saíram:
quarters de estados/parques/2021, quarters com a frente de 1999–2021 (separados), moedas ilegíveis
(escuras/borradas/cobertas) e 2 recortes de cédula marcados como penny. O critério "ilegível" foi
visual. A planilha `moedas.csv` (no zip original) registra cada recorte, a foto de origem e o motivo de
exclusão. **Há várias fotos da mesma moeda em sequência**; 266 imagens ≠ 266 moedas. Os nomes das fotos
de origem começam com o código da sessão (s0a, s0b, s1a, s1b, s2a, s2b, s3a, s3b, s4a). Uma divisão por
sessão foi proposta (treino s0a+s1a+s2a+s3b = 122 L/109 J/19 W; teste o resto = 191/157/22) mas **não foi usada**.

**Erros já cometidos (não repetir):**
- Dizer que o World Coins "não tem moedas americanas": **tem** (pastas 206–211).
- Dizer que as séries do U.S. Coins são "a mesma moeda": os grupos pelo número do nome do arquivo são
  **séries/fontes de fotos** (ex.: um par conferido era um níquel de 1958 e outro de 1942), não a mesma moeda.
- O dataset do Kaggle U.S. Coins publicado tem **só essas 3 classes** (a descrição fala em 29.473 imagens de
  44 categorias, mas isso é a coleção do autor, não a versão publicada).

---

## 5. Resultados, organizados por QP

### QP1 – validação × fora do domínio

| Conjunto | Resultado |
|---|---|
| Validação (U.S. Coins, 749) | **95,1%** (712/749), F1 macro 0,95. Revocação: Jefferson 96,2%, Lincoln 94,0%, Washington 94,9% |
| Fotos boas de outra fonte (`controle_eua`, 50) | **90,0%** (45/50). Jefferson 22/24 (91,7%, confiança média 0,98; 2 → Lincoln); Lincoln 23/26 (88,5%, confiança 0,94; 1 → Jefferson, 2 → Washington) |
| Fotos de celular (620) | **53,1% geral, 51,4% balanceada**. Jefferson 155/266 (58,3%), Lincoln 155/313 (49,5%), Washington 19/41 (46,3%) |

- Matriz da validação (linhas = verdadeira; J, L, W): [[229,4,5],[11,221,3],[11,3,262]]. 22 dos 37 erros
  vão para Jefferson. 6 dos 37 erros têm confiança > 0,894; o pior: Jefferson → Washington com 98,8%.
- Matriz do celular (modelo atual): [[155,4,107],[77,155,81],[21,1,19]]. Principal confusão: Jefferson →
  Washington (107); Lincoln se divide entre as duas (perde a cor); o modelo responde "Lincoln" só 160 vezes.
- **Vazamento** (`check_leakage.py`): agrupando pelo número no nome do arquivo (ex.: "37633" em
  `Lincoln Cent 37633_208 Obverse.jpg`): 290 grupos, média 12,9 imagens, máximo 330. **96% (719/749) da
  validação** tem fotos da mesma série no treino (Jefferson 100%, Lincoln 100%, Washington 89,1%).
  Acurácia 94,9% nessas 719 contra 100% nas outras 30.
- **Duplicatas exatas** (`checar_duplicatas.py`, hash MD5): 16 grupos de arquivos idênticos com nomes
  diferentes no `raw`; **26 das 749 imagens de validação (3,5%) têm cópia exata no treino** (20 Washington,
  ex.: `Washington Quarter 27 25_3.jpg`, `...26 25_3.jpg`). Sem elas, acurácia ≥ 94,9% (pior caso 686/723).
- Experimento antigo de aumento (também mostra validação otimista): completo 95,5%, sem espelhamento 94,7%,
  sem aumento **97,5%** (uma execução cada).

### QP2 – atalhos

**Oclusão e cor** (`occlusion_test.py`, modelo final, 749 imagens de validação). Moeda aproximada por
círculo central de raio 0,45 do lado; centro = 60% desse raio; borda = anel 60–100%; fundo = fora do
círculo; preenchimento cinza 128; e conversão para tons de cinza.

| Teste | Jefferson | Lincoln | Washington | Todas |
|---|---|---|---|---|
| Original | 96,2% | 94,0% | 94,9% | 95,1% |
| Centro coberto | 98,7% | 87,2% | 84,1% | 89,7% |
| Borda coberta | 97,5% | 54,5% | 57,6% | 69,3% |
| Fundo coberto | 97,1% | 91,9% | 29,0% | 70,4% |
| Tons de cinza | 97,9% | 0,4% | 88,0% | 63,7% |

- Lincoln em cinza: 115 → Jefferson, 119 → Washington (depende da **cor acobreada**).
- Washington com fundo coberto: 178 → Jefferson (fundo e/ou borda serrilhada externa; o teste não separa).
- Jefferson fica 96–99% em tudo porque é a **classe padrão** para onde os erros vão, não por robustez.
- Confiança média na classe correta: 0,942 / 0,885 / 0,680 / 0,716 / 0,631.
- "Todas" = acurácia sobre as 749 (não é média das colunas).

**Classe `outra_moeda`** (antigo "Experimento E"; scripts em `testeExperimentoE/`):
- 4ª classe com **1.000 imagens do World Coins** (sorteio equilibrado por tipo, semente 42), excluindo os
  países usados no teste (Canadá, Japão, Reino Unido, Dinamarca), Austrália e EUA. Mesma arquitetura e
  protocolo; divisão 80/20 refeita sobre o novo conjunto (a validação não é idêntica à original).
- **Sem recorte** (28 épocas, melhor 23): acurácia nas conhecidas 94,3%, alarme falso 1,8%, `outra_moeda`
  da validação reconhecida 89,1%; rejeição próximas **88,0%** (AUROC 0,98; Canadá 1c 96,2, 25c 88,5,
  5c 78,8, Japão 100 ienes 89,3), distantes **97,8%** (AUROC 1,00). **Mas no controle**: Jefferson 12,5%
  de acerto (83,3% → outra_moeda), Lincoln 19,2% (80,8% → outra_moeda): **82% das moedas americanas
  conhecidas rejeitadas** → o modelo aprendeu a **fonte da foto** (uCoin), não a moeda.
- **Com localização OpenCV + máscara** (31 épocas, melhor 26): conhecidas 94,1%, alarme falso 2,2%,
  outra_moeda da validação 80,7%; rejeição próximas 60,9% (AUROC 0,92; Canadá 1c 28,8 – acobreada, 25c
  73,1, 5c 63,5, Japão 92,9), distantes 62,0% (AUROC 0,94; Dinamarca 50,0, 2 libras 94,2, 20p 40,4,
  50p 57,7). Controle: Jefferson 45,8% (41,7% → outra), Lincoln 26,9% (53,8% → outra) = 36% de acerto e
  48% rejeitadas. A máscara reduz o atalho, mas não elimina (sobram luz, cor, câmera).
- Localização: `cv2.HoughCircles` (imagem reduzida a 400 px no maior lado, cinza, mediana 5, dp 1,2,
  param1 100, param2 30, raio 20–55% do menor lado, círculo mais forte), recorte quadrado com margem 1,08
  do raio, pixels fora do círculo = cinza 128; sem círculo → quadrado central. Círculo encontrado em 100%
  do controle.
- **AUROC com controle** (desconhecidas × moedas americanas do controle, mesma fonte; `auroc_justo.py`):

| Modelo | Controle rejeitado | Rej. próximas | Rej. distantes | AUROC próximas | AUROC distantes |
|---|---|---|---|---|---|
| Original (confiança 1−c) | – | – | – | 0,53 | 0,85 |
| outra_moeda | 82,0% | 88,0% | 97,8% | 0,62 | 0,67 |
| outra_moeda + máscara | 48,0% | 60,9% | 62,0% | 0,58 | 0,61 |

**Rejeição por protótipos (fora do artigo, só referência):** com moedas próximas, AUROC confiança 0,63,
distância 0,51, combinado 0,50; rejeição a ~10% de alarme falso: 13,6% / 2,2% / 13,0%; regra do protótipo
0,5% (alarme 0,1%). Distantes: AUROC 0,88 / 0,62 / 0,69; rejeição 50,0% / 1,1% / 46,7%; regra 0,0%. Por
tipo (combinado): 50p 65,4; 20p 44,2; 2 libras 42,3; coroa 25,0; Canadá 1c 5,8; 25c 5,8; 5c 19,2;
100 ienes 28,6; (excluídas: Austrália 50c 8,3; Japão 5 ienes 15,4).

### QP3 – degradações (a fazer; Miguel)

Scripts prontos e testados (com modelos falsos), **sem treino**:
- `degradacoes_validacao.py` (158 linhas): aplica às 749 imagens de validação (224×224, 0–255):
  **desfoque** gaussiano σ ∈ {0,5; 1; 1,5; 2; 3; 4}; **luz amarelada** R×(1+a), B×(1−a), a ∈ {0,1…0,5};
  **escurecimento** ×{0,8; 0,6; 0,4; 0,3; 0,2}; **baixa resolução** reduz para {112, 64, 48, 32, 24} px
  e volta a 224. Avalia os 5 modelos (pula os ausentes). Saída `artifacts/experiments/degradacoes/`:
  `degradacoes.csv`, `curvas_degradacoes.png`, `exemplo_degradacoes.png`. ~10–20 min.
- `fatores_celular.py` (102 linhas; importa `degradacoes_validacao.py`): para as 620 fotos de celular
  mede **nitidez** (variância do Laplaciano), **brilho** (média do cinza) e **tom** (média R − média B);
  divide em terços **dentro de cada classe** (o Lincoln é naturalmente avermelhado); reporta acurácia
  balanceada por terço e **% das prateadas classificadas como Lincoln**. Saída
  `artifacts/experiments/fatores_celular/` (`por_imagem.csv`, `resumo.json`).
- Hipóteses registradas antes: luz amarelada derruba as prateadas (viram Lincoln) e o `luz_cor` resiste
  melhor; o `forte` (treinado com desfoque) resiste melhor ao desfoque; o `sem_aumento` cai mais rápido.
  Parte B mostra associação, não causa.
- Checagem: a linha do modelo `atual` na imagem original deve dar **95,1%**; se não, `dataset/raw` difere.
- Guia: `GUIA_MIGUEL_degradacoes.md`.
- Para o Miguel rodar ele precisa de: `dataset/raw` **da Kamilla** (a divisão depende da lista exata de
  arquivos), `dataset/celular/teste_celular`, `artifacts/coin_classifier.keras`,
  `artifacts/class_names.json`, `artifacts/experiments/celular/modelos/*.keras`, os 2 scripts e o repo.
  Em determinado momento ela mandou só o `teste_celular` — conferir se ele recebeu o resto.

### QP4 – aumento de dados (`treinar_aumento.py`, `testar_celular.py`)

Lista **fixa** de variantes definida antes dos resultados; mesma arquitetura e protocolo, só troca as
camadas `aug_*` (inseridas após o `Rescaling`); semente 42; **uma execução cada**.

| Variante | Aumento |
|---|---|
| atual | espelho, rotação 0,05, zoom 0,08, contraste 0,08 (modelo final, não retreinado) |
| sem_aumento | nenhum |
| rotacao | espelho, **rotação 0,5 (±180°)**, zoom 0,08, contraste 0,08 |
| luz_cor | espelho, rotação 0,05, zoom 0,08 + brilho 0,3, contraste 0,3, saturação 0,3, matiz 0,05, cinza p=0,2 |
| forte | espelho, rotação 0,5, zoom 0,2 + tudo do luz_cor + desfoque gaussiano (fator 0,5, núcleo 5, σ 0,1–1,5) |

| Variante | Validação | Celular geral | **Celular balanceada** | Jefferson | Lincoln | Washington |
|---|---|---|---|---|---|---|
| sem_aumento | 98,3 | 42,7 | 45,4 | 33,1 | 49,5 | 53,7 |
| atual | 95,1 | 53,1 | **51,4** | 58,3 | 49,5 | 46,3 |
| rotacao | 93,5 | 44,4 | 45,7 | 45,1 | 43,1 | 48,8 |
| luz_cor | 97,9 | 61,3 | 51,2 | 31,6 | 90,4 | 31,7 |
| forte | 93,5 | 63,9 | 51,5 | 31,2 | 96,5 | 26,8 |

- Matrizes (celular): luz_cor [[84,162,20],[7,283,23],[3,25,13]] (responde Lincoln 470/620);
  forte [[83,172,11],[7,302,4],[5,25,11]] (Lincoln 499/620; 65% dos Jefferson e 61% dos Washington → Lincoln).
- Conclusões: nenhuma variante melhora a **balanceada** (45–52%; diferenças de 1–2 pontos são ruído).
  `sem_aumento` é o melhor na validação e um dos piores no celular. O aumento de **cor** resolve o Lincoln
  (49,5 → 96,5) mas faz o modelo chamar as prateadas de Lincoln; quase acaba com a confusão Jefferson ×
  Washington (quando não diz Lincoln, acerta 81–88%). O efeito vem do aumento de cor (o `luz_cor` não muda
  rotação/zoom e mostra o mesmo). **O atalho troca de classe** (no modelo original a classe padrão era
  Jefferson). A acurácia geral engana (313 Lincoln): **usar sempre a balanceada**.
- Registro: `artifacts/experiments/celular/registro.csv`; JSONs `resultado_<variante>.json`; modelos em
  `artifacts/experiments/celular/modelos/<variante>.keras`.
- Washington tem só 41 imagens (cada uma vale 2,4 pontos; margem ±15). Uma execução por variante.

### Grad-CAM (abandonado)
Feito (`gradcam.py`, `find_examples.py`), a Kamilla não gostou dos resultados e saiu do trabalho. Não
afirmar nada baseado nele.

---

## 6. Scripts e onde estão

| Script | Linhas | Função | Situação no Git |
|---|---|---|---|
| `metrics.py` | – | Acurácia, P/R/F1, matriz, curvas (`artifacts/experiments/exp_a`) | commitado |
| `evaluate_openset.py` | 283 | Rejeição por protótipos (`--unknown-dir`, `--output`) | commitado |
| `occlusion_test.py` | 147 | Oclusão e cor (`artifacts/experiments/oclusao`) | commitado |
| `controle_modelo_original.py` | 37 | Modelo final no `controle_eua` (só imprime; salvar com `> arquivo.txt`) | commit combinado |
| `checar_duplicatas.py` | 49 | Cópias exatas treino × validação | commit combinado |
| `testar_celular.py` | 102 | Avalia um modelo na validação e no celular; anota no `registro.csv` | commit combinado |
| `treinar_aumento.py` | 101 | Treina uma variante (`--variante`, `--epocas`) e chama `testar_celular` | commit combinado |
| `degradacoes_validacao.py` | 158 | QP3 parte A | ainda não está no repo |
| `fatores_celular.py` | 102 | QP3 parte B | ainda não está no repo |
| `check_leakage.py` | 136 | Séries treino × validação (`artifacts/experiments/vazamento`) | no `.gitignore` |
| `train_augmentation.py` | 198 | Experimento antigo de aumento (`exp_b`) | no `.gitignore` |
| `gradcam.py`, `find_examples.py` | 110, 137 | Grad-CAM (abandonado) | no `.gitignore` |
| `make_outra_moeda.py`, `crop_coins.py`, `train_outra_moeda.py`, `auroc_justo.py` | 107, 161, 176, 116 | Classe `outra_moeda` | em `testeExperimentoE/` (ignorado); para rodar de novo, voltar para a raiz |
| `make_group_split.py`, `train_group_split.py` | 91, 114 | Divisão por série | criados, **nunca rodados**; decisão: não fazer |

("commit combinado" = a Kamilla foi orientada a commitar; confirme com `git log` se precisar.)

---

## 7. O artigo (estado e convenções)

- Última versão vista: `TCC___KAMILLA_E_MIGUEL_5.pdf`, 9 páginas, ainda com a estrutura antiga
  (experimentos A–E com letras; rejeição por protótipos como seção). Precisa ser **reorganizado pelas 4 QPs**.
- Estrutura planejada: 1 Introdução (problema, QPs, contribuições) → 2 Referencial (numismática, visão
  computacional, CNNs, aumento de dados, **mudança de domínio, atalhos, vazamento**, transfer learning) →
  3 Trabalhos relacionados (Tabela I: Zaharieva et al. 2007, Anwar et al. 2021, Swain et al. 2023, este
  trabalho; + estudos de robustez/atalhos) → 4 Metodologia → 5 Resultados por QP → 6 Discussão e
  limitações → 7 Conclusão.
- **LaTeX do template (importante):**
  - Tabelas usam **`\tbl{Legenda\label{tab:x}}{\begin{tabular}...\end{tabular}}`** dentro de
    `\begin{table}[H]`. Com `\caption` a legenda sai espremida (uma palavra por linha).
  - Figuras usam `\caption` normal. Figuras/tabelas largas: `figure*`/`table*` com `[t]` (não aceitam `[H]`).
  - Há um ponto duplo nas legendas de figura ("Figura 1. .") vindo do template; causa ainda não achada.
  - A numeração de tabelas pulou a II numa versão (provável `\caption` sobrando).
  - Usar `\textit{}` para nomes de classes; `\%` e vírgula decimal (95,1\%).
- Textos já entregues (podem ser reaproveitados): seções de oclusão (com `tab:oclusao`), validação com
  fotos de outra fonte (com o parágrafo de origem das moedas), metodologia/resultados/discussão/limitações
  do celular + aumento (tabelas `tab:celular` e `tab:matrizes`), parágrafo de limitação com as duplicatas,
  frase sobre campo receptivo/Zeiler e Fergus.
- Pendências conhecidas: reescrever **resumo** (ainda cita experimento de aumento antigo e tem "XX"),
  **introdução** e **referencial** para o novo tema; tirar a seção de rejeição; escrever **Discussão (6)**
  e **Conclusão (7)**; seção 2.8 diz que o desbalanceamento "foi observado" (falso: classes de 1.214 a
  1.285 imagens); remover "RASCUNHO" do topo; 4.4 tinha frase sem base ("O otimizador Adam nos ajudou a
  melhorar...") e "MELHORAR AQUI".

### Referências centrais (reais)
- Geirhos et al. 2020. *Shortcut learning in deep neural networks.* Nature Machine Intelligence 2, 665–673.
- Geirhos et al. 2019. *ImageNet-trained CNNs are biased towards texture...* ICLR.
- Hendrycks e Dietterich 2019. *Benchmarking neural network robustness to common corruptions and perturbations.* ICLR.
- Dodge e Karam 2016. *Understanding how image quality affects deep neural networks.* QoMEX.
- Torralba e Efros 2011. *Unbiased look at dataset bias.* CVPR.
- Recht et al. 2019. *Do ImageNet classifiers generalize to ImageNet?* ICML.
- Kaufman et al. 2012. *Leakage in data mining: formulation, detection, and avoidance.* ACM TKDD 6(4).
- Koh et al. 2021. *WILDS: a benchmark of in-the-wild distribution shifts.* ICML.
- Zeiler e Fergus 2014. *Visualizing and understanding convolutional networks.* ECCV.
- Sandler et al. 2018. *MobileNetV2.* CVPR (só se fizerem a comparação).
- Pech-Pacheco et al. 2000. *Diatom autofocusing in brightfield microscopy.* ICPR (variância do Laplaciano).
- Shorten e Khoshgoftaar 2019 (aumento de dados) e Pan e Yang 2010 (transfer learning) já estão no artigo.
- Datasets: Saharovskiy 2023 (U.S. Coins); wanderdust 2019 (World Coins); Gambill et al. (us-currency-recognition, GitHub; conferir o ano).

---

## 8. Próximos passos (em ordem de prioridade)

1. Repetir `atual`, `sem_aumento` e `forte` com mais 2 sementes (precisa de pequena alteração/novo
   script para aceitar `--semente` sem sobrescrever os modelos existentes) e reportar média ± desvio.
2. QP3 (Miguel): rodar `degradacoes_validacao.py` e `fatores_celular.py`, analisar, escrever.
3. Reescrever o artigo pelas 4 QPs (introdução, referencial, metodologia, resultados, discussão,
   conclusão, resumo), aproveitando os textos já prontos.
4. Mandar ao orientador um resumo curto do escopo (a Kamilla já recebeu modelos de mensagem).
5. (Opcional/futuro) fotos de celular no treino com divisão por sessão; MobileNetV2; test-time augmentation.

---

## 9. Observações finais para o próximo Claude

- O ambiente cloud do Claude Code **não tem** os datasets nem os modelos; não dá para rodar os
  experimentos reais lá. Escreva e teste scripts com dados falsos e peça que eles rodem localmente e
  mandem a saída. (Sugerido a eles: usar o Claude Code local no VS Code para rodar direto.)
- Na rede do ambiente cloud, Kaggle, Roboflow, Hugging Face, Wikimedia, Reddit, eBay e afins são
  bloqueados; GitHub funciona (clone).
- Antes de dar qualquer número ao artigo, confira nesta página ou peça a saída. Antes de afirmar algo
  sobre o que o modelo "aprende", lembre que a evidência é comportamental (oclusão, controles, matrizes).
- Ao sugerir mudanças, prefira o simples. A Kamilla pede "da maneira mais simples possível".
