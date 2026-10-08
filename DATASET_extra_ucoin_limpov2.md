# Conjunto de teste externo `extra_ucoin_limpov2`

Conjunto usado **só para avaliação** (nunca para treino). Tem 78 fotos de moedas americanas tiradas do
dataset público *World Coins*. Este guia permite remontar exatamente o mesmo conjunto.

## 1. Fonte

- **Dataset:** World Coins (Coin Images), por wanderdust, no Kaggle:
  <https://www.kaggle.com/datasets/wanderdust/coin-images>
- **Data de acesso:** 06/10/2026
- A maioria das fotos vem do site uCoin (<https://www.ucoin.net>).
- **Pastas usadas:** só as de moedas americanas, nas três divisões do próprio World Coins
  (`train`, `validation` e `test`). Para este trabalho, todas são teste.

| Pasta do World Coins | Classe do projeto |
|---|---|
| `206` | Lincoln Cents, 1909-Date |
| `207` | Jefferson Nickels, 1938-Date |
| `209` | Washington Quarters, 1932-1998 |

O conjunto **não** usa as pastas `dataset/controle_eua` nem `dataset/extra_ucoin`, que eram versões antigas.

## 2. Regras de seleção

As regras foram definidas olhando só as fotos, **antes** de avaliar qualquer modelo neste conjunto.

1. **Fica** a foto de uma moeda com o **mesmo desenho** da classe do treino:
   - Jefferson: retrato e Monticello originais (1938–2003);
   - Lincoln: retrato clássico, com verso de trigo ou do Lincoln Memorial;
   - Washington: verso com a águia (ou o verso do bicentenário, 1776–1976).
2. **Sai** a foto de **outro tipo de moeda** (Buffalo, Liberty, Shield, Indian Head, Braided Hair,
   Standing Liberty) ou de **outro desenho** (Lincoln com escudo, quarters de estados e parques).
3. **Sai** a foto que **não mostra uma face de uma moeda**: várias moedas juntas, ou frente e verso na mesma imagem.
4. **Sai** a foto que é **cópia exata** (mesmo MD5) de outra do conjunto (fica a primeira, na ordem
   `train` → `validation` → `test`) ou de uma foto do treino (`dataset/raw`).

As regras 1 a 3 foram aplicadas à mão. Cada foto excluída está no arquivo `exclusoes_extra_ucoin_v2.csv`, com o
motivo. A regra 4 é aplicada pelo script.

## 3. Como remontar

1. Baixe o World Coins do Kaggle e descompacte. Ache a pasta que contém `train`, `validation` e `test`
   (algo como `coins/data`).
2. Coloque `montar_extra_ucoin_v2.py` e `exclusoes_extra_ucoin_v2.csv` na pasta do projeto, junto do `config.py`.
3. Rode:
   ```
   python montar_extra_ucoin_v2.py --origem "CAMINHO/coins/data"
   ```
4. O script cria:
   - `dataset/extra_ucoin_limpov2/`: as fotos, uma subpasta por classe;
   - `dataset/extra_ucoin_limpov2_info/manifesto.csv`: todas as 108 fotos candidatas, com o MD5 e o que
     aconteceu com cada uma (usada ou excluída, e o motivo);
   - `dataset/extra_ucoin_limpov2_info/revisao_*.jpg`: todas as candidatas numeradas, com as excluídas em vermelho.

## 4. Resultado esperado

| Moeda | Candidatas | Excluídas à mão | Cópias exatas | **Usadas** |
|---|---|---|---|---|
| Jefferson | 36 | 6 | 2 | **28** |
| Lincoln | 36 | 4 | 12 | **20** |
| Washington | 36 | 4 | 2 | **30** |
| **Total** | **108** | **14** | **16** | **78** |

(Confira com a tabela que o script imprime no final.)

## 5. Como conferir que é o mesmo conjunto

O manifesto da versão usada no trabalho está no repositório, em `manifesto_extra_ucoin_limpov2.csv`. Depois de
remontar, compare os dois arquivos:

```
fc.exe manifesto_extra_ucoin_limpov2.csv dataset\extra_ucoin_limpov2_info\manifesto.csv
```

Se aparecer "nenhuma diferença encontrada", o conjunto é idêntico, foto por foto (o MD5 de cada arquivo está
no manifesto).

## 6. Como avaliar um modelo

```
python avaliar_pasta.py --modelo CAMINHO/modelo_divisao_series.keras --pasta dataset/extra_ucoin_limpov2 --nome NOME
```

