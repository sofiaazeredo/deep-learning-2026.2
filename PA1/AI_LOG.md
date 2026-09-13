# AI_LOG

Uso de IA no PA1.

> **Para a dupla preencher.** O que está abaixo é o registro factual dos
> episódios em que a IA foi usada na segunda metade do trabalho (Claude Code,
> via terminal, sobre este repositório). Falta a primeira metade — a construção
> do dataset, do modelo, das perdas, da métrica e das ablações de perda — que foi
> feita por vocês e cujo relato de uso de IA só vocês podem escrever.
>
> A política do PDF é clara: o que não vale é entregar algo que vocês não
> entendem. Revisem cada item, corrijam o que estiver errado e apaguem este
> aviso.

## Como a IA foi usada

Ferramenta: Claude Code (Anthropic), rodando localmente com acesso ao
repositório. Os treinos rodaram no Google Colab (Tesla T4); a IA não teve acesso
ao Colab e não executou nenhum treinamento.

### Episódio 1 — template do notebook a partir do PDF

Pedimos a leitura do enunciado e a geração de um notebook com uma célula por
questão, para organizar o trabalho. Serviu como esqueleto inicial.

### Episódio 2 — ambiente

Instalação do PyTorch com CUDA e das bibliotecas permitidas, checagem de que
`scipy.ndimage`, `skimage.segmentation` e `sklearn.cluster` importavam, e
geração do `requirements.txt`.

### Episódio 3 — bug que só apareceria na ablação de resolução

`scripts/evaluate_boundary.py` referenciava `UNetNoSkips` e `UNetASPP` sem
importar. Como todas as avaliações anteriores usaram `unet`, o ramo nunca tinha
sido executado e o erro só apareceria ao avaliar os checkpoints novos. A correção
centralizou a reconstrução do modelo em `src.model.load_model_from_checkpoint`,
que lê a arquitetura gravada no próprio checkpoint.

### Episódio 4 — a métrica não escalava

`instance_map` recalculava a matriz de IoU a cada limiar, e os scripts de
avaliação chamavam `instance_precision` mais dez vezes: vinte matrizes por
imagem, cada uma um laço sobre pares de máscaras booleanas de imagem inteira. No
mosaico da Parte 4 isso dava 30 s por matriz e travava a análise. Reescrito com
`np.bincount`; verificado que o resultado é idêntico ao anterior (diferença
máxima 0,00) e ~1500× mais rápido.

### Episódio 5 — `ModuleNotFoundError: No module named 'src'` no Colab

`python scripts/x.py` coloca `scripts/` no `sys.path`, não a raiz do projeto.
Funcionava numa imagem do Colab e parou de funcionar em outra. Cada script passou
a inserir a raiz no `sys.path`.

### Episódio 6 — o diagnóstico da Parte 5

Este é o episódio em que a IA encontrou algo que não estávamos procurando.

Na pior imagem do teste o modelo previa **zero** núcleos contra 78 no ground
truth, mas o mapa intermediário de fronteira mostrava um anel nítido em volta de
cada núcleo — a rede tinha encontrado todos. A investigação mostrou a causa em
`src/dataset.py`: o alvo de 3 classes é construído com
`binary_erosion(mask, iterations=2)`, e um núcleo de 4 px de diâmetro perde o
interior inteiro. Naquela imagem, 63% dos núcleos ficam sem nenhum pixel de
interior no alvo (91 pixels de interior contra 1106 de fronteira). No dataset
inteiro: 100% dos núcleos com diâmetro ≤ 4 px e 29,8% dos de 4–8 px. Sem
interior não há marcador para o watershed, e o objeto não pode ser emitido.

A correção de pós-processamento sugerida pelo diagnóstico (semear um marcador no
ponto mais interno de cada componente sem marcador) foi implementada e
**piorou** o mAP médio em 0,021, ainda que resolvesse o caso alvo. Isso está
reportado como está, porque é o que o PDF pede quando a correção não funciona.
A correção que de fato resolve é no alvo (`--adaptive-boundary`).

### Episódio 7 — Partes 4, 6 e 7

Implementação do driver de inferência em mosaico com as três decodificações e o
mAP antes/depois da fusão, do teste de estresse (corrupções e escala) e do
pipeline de inferência (`src/inference.py` + `inferencia.ipynb`).

### Episódio 8 — campo receptivo

Cálculo camada a camada, com e sem atrous, comparado com a distribuição de
tamanhos dos núcleos. Resultado: 200 px sem ASPP, 456 px com, contra núcleos de
12 px de mediana e 83 px no máximo — o campo receptivo não é o gargalo deste
dataset, ao contrário do exemplo do enunciado.

## O que a IA NÃO fez

- Não rodou nenhum treinamento; todos foram executados por nós no Colab.
- Não escolheu a trilha da Parte 2 nem os eixos da Parte 3.
- Não escreveu a apresentação.

## Erros da IA que tivemos que corrigir

> **Preencher.** Houve pelo menos estes, registrados na sessão:
>
> - Numa verificação, a IA executou `scripts/train.py` sem querer (o script não
>   tem `--help`, então rodar para "testar o import" dispara um treino) e
>   sobrescreveu `experiments/results/training_history.csv`. Foi restaurado do
>   git.
> - Uma célula de setup sugerida clonava o repositório de novo por cima de si
>   mesmo, criando `PA1/deep-learning-2026.2/` na branch errada.
> - Acrescentem os casos em que a sugestão estava errada e vocês perceberam.
