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
| 1 — baseline semântico + componentes conexos | U-Net, 2 classes | 0,438 | 13,10 |
| 2 — fronteira + watershed (Trilha A) | U-Net, 3 classes | 0,489 | 12,24 |

**Parte 3, Eixo 1 — como recuperar resolução** (média ± desvio, 2 seeds, mesma
perda e mesmo split):

| Arquitetura | mAP | Erro de contagem |
|---|---|---|
| U-Net + skips | **0,4885 ± 0,0070** | 12,24 ± 0,30 |
| U-Net + ASPP | 0,4424 ± 0,0208 | 12,80 ± 0,69 |
| U-Net sem skips | 0,2699 ± 0,0770 | 25,01 ± 7,98 |

**Parte 3, Eixo 2 — função de perda** (2 seeds):

| Perda | mAP |
|---|---|
| Weighted CE | 0,4885 ± 0,0070 |
| Focal γ=1 | 0,4867 ± 0,0071 |
| CE | 0,4826 ± 0,0052 |
| Focal γ=2 | 0,4816 ± 0,0060 |
| Focal γ=5 | 0,3716 ± 0,0392 |

Modelo final: **U-Net com skip connections + Weighted CE**, seed 123.

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

O modelo final (`loss_balanced_seed123_best.pt`, 119 MB) não cabe no limite de
arquivo do GitHub e está aqui:

**<COLAR O LINK DO DRIVE AQUI>**

Baixe para `checkpoints/` antes de rodar as Partes 4 a 7.

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
- **Split.** `create_splits` usa `random_split` com seed fixa. **Não é
  estratificado por modalidade**, que é o que o PDF pede; as consequências
  aparecem na Parte 5 (o mAP por imagem correlaciona −0,64 com o brilho, isto é,
  o modelo vai mal justamente nas modalidades claras, minoritárias no treino).
- **Resolução.** Todas as imagens são redimensionadas para 256×256 em
  `__getitem__`, inclusive as de 1024×1024 e 1040×1388. As métricas são medidas
  nessa resolução, não na original.
