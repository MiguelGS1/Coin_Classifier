# Coin_Classifier

## 1. Baixar o projeto

Clone o repositório:

git clone URL_DO_REPOSITORIO
cd Coin_Classifier

Ou baixe o .zip pelo GitHub e abra a pasta do projeto no terminal.

## 2. Criar e ativar o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate
## 3. Instalar as dependências
pip install --upgrade pip
pip install -r requirements.txt
## 4. Baixar e organizar o dataset

O dataset completo não está incluído no repositório por causa do tamanho.

Baixe o dataset em:

https://www.kaggle.com/datasets/sergiosaharovskiy/uscoins

Depois, coloque as imagens dentro da pasta:

dataset/raw/

A estrutura deve ficar assim:

dataset/raw/
  1_euro/
  1_real_brasil/
  Jefferson Nickels, 1938-Date/
  Lincoln Cents, 1909-Date/
  Washington Quarters, 1932-1998/

Cada pasta representa uma classe de moeda.

## 5. Treinar o modelo

Depois de organizar o dataset, rode:

python3 train.py

Ao final do treinamento, será criada a pasta artifacts/, contendo o modelo treinado e os arquivos necessários para fazer previsões.

## 6. Testar uma imagem

Depois que o treinamento terminar, rode:

python3 predict.py --image "caminho/da/imagem.jpg"

Exemplo usando uma imagem do dataset:

python3 predict.py --image "dataset/raw/Jefferson Nickels, 1938-Date/Jefferson Nickel 26324_109 Reverse.jpg"

O programa retornará as porcentagens de chance para cada classe de moeda e também para outra_moeda.

## 7. Rodar novamente depois

Se o modelo já foi treinado, não precisa rodar train.py novamente.

Basta ativar o ambiente virtual e usar o predict.py:

source .venv/bin/activate
python3 predict.py --image "caminho/da/imagem.jpg"
Quando rodar train.py novamente?

Rode train.py novamente apenas se:

adicionar novas imagens ao dataset;
remover imagens do dataset;
adicionar uma nova classe de moeda;
remover uma classe de moeda;
apagar a pasta artifacts/;
alterar configurações do modelo.