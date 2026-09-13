# PA1 — Segmentação de instâncias sem detector

Aprendizado Profundo (FGV) — Programming Assignment 1.

Fazer as arquiteturas da aula de segmentação **semântica** produzirem rótulos
**instance-aware**, sem Mask R-CNN nem qualquer detector com proposta de região.

Dataset: **DSB2018 / BBBC038v1** (Opção A), 670 imagens de microscopia com uma
máscara PNG por núcleo.

---

## Resultados

| Parte | Modelo | mAP@0,50:0,95 | Erro de contagem |
|---|---|---|---|
| 0 — teste unitário sintético | U-Net reduzida, 3 classes | 0,523 | 1,41 |
| 1 — baseline semântico + componentes conexos | U-Net, 2 classes | 0,422 | 8,68 |
| 2 — fronteira + watershed (Trilha A) | U-Net, 3 classes | 0,504 | 8,44 |

Modelo final (`resolution_aspp_seed42`) no teste: **mAP 0,4911**, erro de contagem 8,08.

Todos os números abaixo usam o **split estratificado por modalidade** (66 imagens
de teste).

**Parte 0 — teste unitário sintético**: 256 imagens 128×128 com 5 a 20 elipses
(13,2 por imagem em média, 84,3% encostando em algum vizinho), ruído e contraste
variáveis. Treina em **126 s (2,1 min)** numa T4, dentro do limite de 5 minutos
do enunciado, e atinge mAP 0,523. Usa o mesmo encoder-decoder, a mesma perda, a
mesma decodificação por watershed e a mesma métrica dos dados reais — se alguma
peça do pipeline quebrar, quebra aqui em minutos.

**Parte 3, Eixo 1 — como recuperar resolução** (média ± desvio, 2 seeds, mesma
perda e mesmo split):

| Arquitetura | mAP | Erro de contagem |
|---|---|---|
| U-Net + ASPP | 0,4834 ± 0,0108 | 8,32 |
| U-Net + skips | 0,4770 ± 0,0094 | 8,57 |
| U-Net sem skips | 0,3697 ± 0,0228 | 12,04 |

**A diferença entre ASPP e skips NÃO é conclusiva**: 0,0065 de distância contra
0,0202 de soma dos desvios, com 2 seeds. O que o experimento mostra com clareza é
o terceiro lugar — **tirar as skip connections custa 0,11 de mAP e 45% a mais de
erro de contagem**. Entre skips e ASPP, o experimento não decide.

As três arquiteturas foram treinadas com `balanced_ce` para a comparação ficar
controlada, embora o Eixo 2 mostre que essa não é a melhor perda.

**Parte 3, Eixo 2 — função de perda** (2 seeds):

| Perda | mAP | Erro de contagem |
|---|---|---|
| CE | **0,5040 ± 0,0048** | 8,44 |
| Focal γ=2 | 0,5033 ± 0,0055 | 8,59 |
| Focal γ=1 | 0,4949 ± 0,0054 | 8,74 |
| Weighted CE | 0,4770 ± 0,0094 | 8,57 |
| Focal γ=5 | 0,4350 ± 0,0343 | 8,56 |

CE, focal γ=2 e focal γ=1 estão dentro de um desvio umas das outras. O que separa
é o extremo: **γ=5 desaba** (0,435, com desvio 7× maior que os demais), e pesar a
classe fronteira (Weighted CE) **piora** em vez de ajudar.

Modelo final: **U-Net + ASPP**, seed 42.

**Parte 4 — inferência em mosaico** (mosaico 3×3, tiles de 256 px, sobreposição 64 px):

| decodificação | mAP | núcleos previstos |
|---|---|---|
| passada única | 0,4585 | 290 |
| tiles, sem fusão | 0,3548 | 413 |
| tiles, com fusão | **0,4510** | 306 |
| ground truth | — | 337 |

A fusão de instâncias entre tiles recupera **+0,0961 de mAP** e elimina os ~107
objetos duplicados nas costuras, voltando praticamente ao nível da passada única.

**Parte 5 — campo receptivo**: 200 px sem ASPP, 456 px com. Os núcleos têm 12 px
de mediana e 83 px no máximo: **0,00% excedem o campo receptivo**, ou seja, o
campo receptivo não é o gargalo deste dataset.

**Parte 6 — teste de estresse** (corrupções, 3 intensidades):

Referência sem estresse: mAP 0,4911.

| | leve | média | forte |
|---|---|---|---|
| blur (σ) | 0,4408 (−10%) | 0,2916 (−41%) | 0,1154 (−77%) |
| ruído (σ) | 0,3744 (−24%) | 0,1132 (−77%) | 0,0135 (−97%) |
| brilho (Δ) | 0,1221 (−75%) | 0,0049 (−99%) | 0,0000 (−100%) |

Escala: 0,5× → 0,2787 (−43%), 1× → 0,4911, 2× → 0,3586 (−27%).

Brilho é de longe o eixo mais destrutivo: um deslocamento de 0,30 já derruba 99%
do mAP. A degradação com a escala é **assimétrica** — reduzir pela metade custa
bem mais que dobrar.

---

## Ambiente

Python 3.11+ com PyTorch. No Colab não é preciso instalar nada além do que já vem.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

Os scripts inserem a raiz do projeto no `sys.path` sozinhos, então rodam de
qualquer diretório sem `PYTHONPATH`.

## Dados

Baixe o `stage1_train` do DSB2018 e coloque em `PA1/data/raw/`, uma pasta por
amostra:

```
data/raw/<sample_id>/images/<sample_id>.png
data/raw/<sample_id>/masks/*.png          # um PNG por núcleo
```

Fonte: <https://bbbc.broadinstitute.org/BBBC038> (download direto, sem conta) ou
`kaggle competitions download -c data-science-bowl-2018`.

Os dados já estão versionados neste repositório, então um `git clone` basta.

## Um comando que treina

```bash
python scripts/train_boundary.py --architecture unet --loss balanced_ce --seed 123 --epochs 20 --name modelo_final
```

## Um comando que avalia

```bash
python scripts/evaluate_boundary.py --checkpoint checkpoints/modelo_final_best.pt --name modelo_final
```

## Inferência numa imagem qualquer

Sem retreinar, aceita qualquer tamanho, RGB/RGBA/tons de cinza:

```python
from src.inference import segment_image

result = segment_image("caminho/para/imagem.png")
print(result["count"])     # número de núcleos
result["colored"]          # máscara de instâncias colorida
```

O notebook `inferencia.ipynb` faz o mesmo com figuras.

---

## Reproduzir cada parte

O split é fixo (`SPLIT_SEED = 42`) e independente da seed de treino, então todas
as execuções compartilham exatamente o mesmo conjunto de teste.

```bash
# Parte 0 — teste unitário sintético (não precisa dos dados reais)
python scripts/train_synthetic.py

# Parte 1 — baseline semântico + instâncias ingênuas
python scripts/train.py
python scripts/evaluate.py
python scripts/plot_baseline_results.py

# Parte 2 — fronteira + watershed
python scripts/train_boundary.py --loss balanced_ce --seed 42 --epochs 20 --name boundary
python scripts/evaluate_boundary.py --checkpoint checkpoints/boundary_best.pt --name boundary

# Parte 3, Eixo 1 — recuperação de resolução (2 seeds por arquitetura)
for arch in no_skips aspp; do
  for seed in 42 123; do
    python scripts/train_boundary.py --architecture $arch --loss balanced_ce \
      --seed $seed --epochs 20 --name resolution_${arch}_seed${seed}
    python scripts/evaluate_boundary.py \
      --checkpoint checkpoints/resolution_${arch}_seed${seed}_best.pt \
      --name resolution_${arch}_seed${seed}
  done
done
python scripts/summarize_resolution_ablation.py

# Parte 3, Eixo 2 — função de perda
python scripts/summarize_loss_ablation.py

# Parte 4 — inferência em mosaico
python scripts/mosaic_inference.py --grid 3 --name mosaic

# Parte 5 — campo receptivo, galeria de falhas e correção
python scripts/receptive_field.py
python scripts/failure_gallery.py --n-failures 5 --name failures
python scripts/failure_correction.py --name correcao

# Parte 6 — teste de estresse
python scripts/stress_test.py --mode corruptions
python scripts/stress_test.py --mode scale
```

A partir da Parte 4 os scripts leem `experiments/results/best_architecture.json`
e resolvem o checkpoint sozinhos; `--checkpoint` força outro.

## Checkpoint

O modelo final (`resolution_aspp_seed42_best.pt`, 151 MB) não cabe no limite de
arquivo do GitHub (100 MB) e está no Drive:

**<COLAR O NOVO LINK DO DRIVE AQUI>**

Baixe para `PA1/checkpoints/` antes de rodar as Partes 4 a 7:

```bash
mkdir -p checkpoints
# baixe o arquivo do link acima para checkpoints/resolution_aspp_seed42_best.pt
python scripts/mosaic_inference.py --grid 3 --name mosaic   # já encontra o checkpoint sozinho
```

É a U-Net com ASPP treinada com `balanced_ce`, seed 42 — a vencedora do Eixo 1
(por uma margem dentro do ruído, veja acima). Todas as figuras das Partes 4, 5 e
6 saem dele.

---

## Estrutura

```
src/
  dataset.py          DSB2018Dataset, split, alvo de 3 classes (fronteira)
  model.py            UNet, UNetNoSkips, UNetASPP + registro e loader de checkpoint
  losses.py           CE, CE balanceada, focal
  metrics.py          IoU, Dice e mAP de instância implementado do zero
  postprocessing.py   limiar+componentes conexos, fronteira+watershed
  mosaic.py           tiles sobrepostos e fusão de instâncias entre tiles
  inference.py        imagem qualquer -> máscara colorida + contagem
  synthetic.py        gerador de elipses sintéticas (Parte 0)
scripts/              um script por etapa (ver acima)
experiments/
  results/            um CSV por execução + os sumários das ablações
  figures/            todas as figuras da apresentação
inferencia.ipynb      entregável de inferência, roda sem retreinar
AI_LOG.md             uso de IA neste trabalho
```

## Notas de método

- **Métrica de instância.** `src/metrics.py` implementa o matching à mão, como o
  PDF exige. Casamento **guloso por IoU decrescente**, um-para-um, e
  AP = TP / (TP + FP + FN) por limiar, de 0,50 a 0,95 em passos de 0,05. A matriz
  de IoU sai de um `np.bincount` sobre os pares de rótulos, e não de um laço por
  par de máscaras — o resultado é idêntico e viabiliza a escala do mosaico.
- **Split.** `create_splits` é **estratificado por modalidade**, como o enunciado
  exige. O DSB2018 não traz rótulo de modalidade, então derivamos um de duas
  estatísticas da imagem: separação média de canais (`max−min` entre RGB) e
  brilho médio. Os cortes caem em vazios do histograma, não no meio de nada — a
  separação de canais é exatamente 0,0 em 562 das 670 imagens e ≥ 30 nas outras
  108; entre as acinzentadas o brilho é ≤ 57 ou ≥ 205. Um KMeans(k=4)
  independente reencontra os mesmos três grupos: `fluor_escura` (546),
  `histologia_colorida` (108) e `brightfield_clara` (16). `split_report` imprime
  a distribuição por split. `stratify=False` reproduz o `random_split` anterior.
- **Sobre o brilho.** O mAP por imagem correlaciona ≈ −0,61 com o brilho. Isso
  **não** se explica por falta de imagens claras no treino: elas são ~18% do
  treino nos dois splits (o sorteio aleatório já tinha caído balanceado, o que
  medimos). As modalidades claras são simplesmente mais difíceis — a pior imagem
  do teste é citologia em campo claro com núcleos de 4,4 px.
- **Resolução.** Todas as imagens são redimensionadas para 256×256 em
  `__getitem__`, inclusive as de 1024×1024 e 1040×1388. As métricas são medidas
  nessa resolução, não na original.
