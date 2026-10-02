# Guia – Parte do Miguel: QP3 (degradações)

## O trabalho

**Título:** *Atalhos de aprendizado e generalização fora do domínio em redes convolucionais: um estudo de
caso com moedas*

| QP | Pergunta | Papel | Situação |
|---|---|---|---|
| QP1 | A validação comum superestima o desempenho fora do domínio? Quanto? | Mostrar o problema | Feito: 95,1% (validação) → 90,0% (fotos boas de outra fonte) → 51,4% (celular, balanceada) |
| QP2 | Em que o modelo se apoia? Ele aprende atalhos? | Explicar | Feito: cor (Lincoln), fundo (Washington), fonte da foto (classe outra_moeda) |
| **QP3** | **Que tipo de degradação mais prejudica o modelo: mudança de cor ou perda de detalhe?** | **Testar a explicação** | **Sua parte** |
| QP4 | O aumento de dados reduz a perda? | Tentar corrigir | Feito: nenhuma variante melhorou (~51%); o atalho só troca de classe |

**Por que a QP3:** as fotos de celular misturam vários problemas ao mesmo tempo (luz, cor, desfoque,
resolução). A QP3 separa cada um, com degradações controladas, e responde se a queda vem mais do
**atalho de cor** (QP2) ou da **perda de detalhe** (desfoque, baixa resolução). Método inspirado em
Hendrycks e Dietterich (2019).

**Nenhum treino é necessário.** Os dois scripts só avaliam os 5 modelos que já existem.

## O que precisa estar no computador

- Projeto com `dataset/raw` (U.S. Coins) e `dataset/celular/teste_celular` (620 fotos de celular).
- Os 5 modelos: `artifacts/coin_classifier.keras` (atual), `artifacts/class_names.json` e
  `artifacts/experiments/celular/modelos/` com `sem_aumento`, `rotacao`, `luz_cor` e `forte` (.keras).
- OpenCV (`python -m pip install opencv-python`). Rodar com o Python do sistema, sem `.venv`.
- Os dois scripts na raiz do projeto: `degradacoes_validacao.py` e `fatores_celular.py`.

Os scripts não alteram nenhum arquivo do projeto. Se algum modelo faltar, eles avisam e seguem com os outros.

## Parte A – Degradações controladas (`degradacoes_validacao.py`)

```powershell
python degradacoes_validacao.py
```

Aplica 4 degradações, em níveis crescentes, às 749 imagens de validação (já em 224×224) e mede a
acurácia dos 5 modelos em cada nível. Leva uns 10 a 20 minutos.

| Fator | Níveis | O que simula | Ligação |
|---|---|---|---|
| desfoque | σ = 0,5 / 1 / 1,5 / 2 / 3 / 4 | Foto tremida ou fora de foco | Perda de detalhe |
| luz amarelada | a = 0,1 a 0,5 (vermelho ×(1+a), azul ×(1−a)) | Luz de lâmpada | **Testa o atalho de cor** |
| escurecimento | brilho × 0,8 / 0,6 / 0,4 / 0,3 / 0,2 | Pouca luz | Cor e detalhe |
| baixa resolução | reduz para 112 / 64 / 48 / 32 / 24 px e volta para 224 | Moeda pequena na foto | Perda de detalhe |

**Saídas** (`artifacts/experiments/degradacoes/`):
- `degradacoes.csv`: modelo × fator × nível, com acurácia geral e por classe;
- `curvas_degradacoes.png`: uma curva por modelo em cada fator (figura principal);
- `exemplo_degradacoes.png`: a mesma moeda em todos os níveis (boa figura para o artigo).

**Hipóteses (escritas antes de rodar):**
- H1: a acurácia cai com todos os fatores, mas a **luz amarelada** derruba principalmente as
  **moedas prateadas** (que passam a parecer cobre) e o atual sofre mais que o `luz_cor`.
- H2: o `forte`, treinado com desfoque, cai menos no desfoque que os outros modelos.
- H3: o `sem_aumento` (melhor na validação) é o que cai mais rápido.

**O que olhar:** além da acurácia geral, as **colunas por classe** do CSV. Se a luz amarelada derrubar
Jefferson e Washington (e não o Lincoln), é forte evidência do atalho de cor.

## Parte B – Fatores nas fotos reais de celular (`fatores_celular.py`)

```powershell
python fatores_celular.py
```

Mede em cada uma das 620 fotos de celular:
- **nitidez** (variância do Laplaciano; menor = mais borrada);
- **brilho** (média dos tons de cinza; menor = mais escura);
- **tom** (média do vermelho − média do azul; maior = mais amarelada).

Divide as fotos em 3 terços para cada medida (baixo, médio, alto) e mostra, para cada modelo, a
**acurácia balanceada** em cada terço e a **% das moedas prateadas classificadas como Lincoln**.

**Importante:** os terços são calculados **dentro de cada classe**. O Lincoln é naturalmente mais
avermelhado, e dividir todas juntas misturaria o efeito do tom com o efeito da classe.

**Saídas** (`artifacts/experiments/fatores_celular/`):
- `por_imagem.csv`: uma linha por foto, com as medidas, os terços e a previsão de cada modelo;
- `resumo.json`: os números da tabela.

**Hipóteses:**
- o acerto é menor nas fotos mais borradas e nas mais escuras;
- nas fotos mais amareladas, mais moedas prateadas são chamadas de Lincoln.

**Cuidado:** a Parte B mostra **associação**, não causa (uma foto borrada também tende a ser escura). A
Parte A, controlada, é a que permite falar em causa. As duas juntas são o argumento.

## O que entregar

1. Tabela da Parte A (pode ser só o modelo atual e o forte, ou os 5) e a figura de curvas.
2. A figura de exemplo das degradações.
3. Tabela da Parte B: modelos × terços, para nitidez, brilho e tom.
4. Um parágrafo de método e um de resultados, com os números reais.

## Como isso fecha a história

- Se a **luz amarelada** derrubar muito as prateadas → confirma o atalho de cor (QP2) por outro caminho.
- Se o **desfoque/resolução** derrubarem muito → o modelo depende de detalhe fino, e fotos de celular
  borradas são parte central do problema.
- Se o `forte` resistir ao desfoque na Parte A mas não melhorar no celular (QP4) → o desfoque não é o
  principal fator das fotos reais.
Qualquer resultado é útil para a discussão.

## Referências

```bibtex
@inproceedings{hendrycks2019benchmarking,
  author    = {Hendrycks, Dan and Dietterich, Thomas},
  title     = {Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2019}
}
@article{geirhos2020shortcut,
  author  = {Geirhos, Robert and Jacobsen, J{\"o}rn-Henrik and Michaelis, Claudio and Zemel, Richard
             and Brendel, Wieland and Bethge, Matthias and Wichmann, Felix A.},
  title   = {Shortcut Learning in Deep Neural Networks},
  journal = {Nature Machine Intelligence},
  volume  = {2},
  pages   = {665--673},
  year    = {2020}
}
@inproceedings{pechpacheco2000diatom,
  author    = {Pech-Pacheco, Jos{\'e} Luis and Crist{\'o}bal, Gabriel and Chamorro-Mart{\'\i}nez, Jes{\'u}s
               and Fern{\'a}ndez-Valdivia, Joaqu{\'\i}n},
  title     = {Diatom Autofocusing in Brightfield Microscopy: A Comparative Study},
  booktitle = {International Conference on Pattern Recognition (ICPR)},
  pages     = {314--317},
  year      = {2000}
}
```
