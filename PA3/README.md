# PA3 — Poucos rótulos (fauna em câmeras trap) · Trilha B (FixMatch)

Classificação de 21 espécies (ENA24) com 5 rótulos/classe + o `train` sem rótulo. Backbone pequeno (≤ 12M parâmetros), roda em CPU.

> **Estado:** pipeline rodado nos dados reais (Colab T4): sweeps, Parte 0, Parte 2, modelo final, galeria e `predict.py`. O notebook `colab/PA3_colab.ipynb` executa tudo do zero; a ablação `corr=none`, os diagnósticos offline e a análise de 3000 passos estão nas seções 7.1–7.4.
> Nenhum valor de treino é fixo no código: sweeps escolhem hiperparâmetros no val.
> **Nota sobre τ:** com `fix.corr=flex` o limiar efetivo é `max(τ·β/(2−β), flex_floor)`; com β pequeno vira o piso e τ deixa de ter efeito. O diagnóstico grava também a versão bruta (`diag_tau_bruto.*`) e há a ablação `none`.

## Ambiente
Python ≥ 3.10, PyTorch/torchvision, `pip install -r requirements.txt`. Colab com GPU (T4) é suficiente. Pesos ImageNet são baixados pelo torchvision
**só durante o treino**; o `predict.py` não baixa nada.

## Dados
Pacote do enunciado (`pa3_dados.zip`), descompactado em `dados/` (ou `/content/dados` no Colab). Layout real, conforme o README do pacote:
`images/{train,val,test_public}/*.jpg`, `labels/{rotulos_1,rotulos_5,rotulos_20,val,test_public}.csv` (colunas `image_id,label`; `image_id` = nome do arquivo sem `.jpg`) e `classes.csv`.
**Não há `train.csv`**: o pool não rotulado (6.505 imagens) é a pasta `images/train/`. O `check_data` detecta isso e grava no schema; `src/common.py` lê o pool da pasta.

Derivado do ENA24-detection (LILA BC, CDLA-Permissive-1.0). Citação pedida pelos autores: Yousif H, Kays R, He Z. *Dynamic Programming Selection of Object Proposals for Sequence-Level Animal Species Classification in the Wild.* IEEE TCSVT, 2019.

## Um comando que treina
```bash
bash scripts/reproduzir.sh dados resultados      # testes → check dos dados → sweeps (val) → Parte 0/2 → modelo final → Parte 3
```
No Colab use `colab/PA3_colab.ipynb` (mesma sequência, célula a célula, com as imagens). Tudo é retomável.

## Um comando que avalia
```bash
python predict.py --imagens PASTA --saida preds.csv          # CSV image_id,label; CPU; imprime nº de parâmetros
python -m src.avaliar --preds preds.csv --gt dados/test_public.csv --data dados   # F1 macro e acurácia
```

## Fluxo e onde cada decisão é tomada
| Etapa | Comando | Decide / produz |
|---|---|---|
| Testes de código | `python -m tests.test_units` | 12 testes (pré-processamento, FlexMatch, alinhamento, EMA, Grad-CAM, checkpoint…) |
| Dados | `python -m tests.check_data --data dados` | confere contagens/orçamentos/splits do enunciado e **detecta o schema** → `configs/data_schema.json` |
| Pipeline | `python -m tests.check_pipeline --data dados` | overfit sem aug, seed, guarda do test_public, FixMatch ×4 correções, **s/passo** |
| Aumentações | `python -m tests.viz_aug --data dados` | você vê se o animal pequeno sobrevive ao recorte |
| Hiperparâmetros | `python -m src.sweep --method {part0_scratch,part0_ft,fixmatch} …` | busca aleatória **só no val** → `configs/selected/*.json` |
| Parte 0 | `python -m src.part0 --data dados` | piso (a) CNN do zero, (b) backbone ImageNet; curva, confusão, época em que decora |
| Parte 1 (B) + 2 | `python -m src.run_final --data dados` | piso e FixMatch nos orçamentos 1/5/20 (3 seeds no 5) → `resultados/curva_orcamento.csv` + diagnósticos de τ e de viés |
| Gráficos | `python -m src.grafico_orcamento` | curva (x em log, teto) + ganho por classe |
| Modelo entregue | `python -m src.select_final --data dados` | escolhe a seed **pelo val** → `pesos/modelo_final.pt` |
| Diagnósticos offline | `python -m src.diag_offline --data dados --runs resultados/runs/fixmatch_b5_s0` | ECE, confiabilidade, τ bruto, erro cinza × colorido (de `test_probs.npy`) |
| Ablação | `python -m src.comparar_ablacao resultados resultados_none_tau095 resultados_none_tau05` | val/test F1 médio±desvio e diferença pareada por seed |
| Parte 3 | `python -m src.galeria --data dados` | 3 erros + 5 acertos com Grad-CAM e oclusão |
| Entrega | `python -m tests.check_predict --data dados` | CPU, rede bloqueada, formato do CSV, == caminho de avaliação, tempo |

## O que NÃO está fixo no código (e onde é decidido)
`configs/defaults/*.json` são **pontos de partida não validados** (marcados no `_status`). Os resultados só usam `configs/selected/*.json`
(sem eles os scripts recusam rodar, a menos que se passe `--allow_defaults`, que fica registrado em `metrics.json`).

| Decisão | Espaço de busca (`configs/sweeps/`) | Como é escolhida |
|---|---|---|
| Backbone (resnet18 / mobilenet_v3_large / efficientnet_b0), `data.size`, `data.resize_mode` (squash × pad) | `part0_ft.json` | val F1 macro do piso, orçamento 5; FixMatch herda o mesmo backbone |
| lr, wd, épocas, escala do crop | `part0_*.json` | val F1 médio entre seeds (desempate: val loss) |
| τ, correção (`none/da/flex/both`), λ, lr, escala do crop, magnitude RandAugment | `fixmatch.json` | val F1 médio entre seeds |
| Época/passo do checkpoint | — | val F1 dentro de cada run |
| Modelo entregue | — | maior val F1 entre as 3 seeds do orçamento 5 |

Constantes do enunciado ficam em `configs/project.json` (21 classes, limite de 12M, teto 0,686, tamanhos esperados dos conjuntos).
Constantes de **algoritmo** que vêm do paper e não são tuning: o cosseno `cos(7πk/16K)` do FixMatch e a função `x/(2−x)` do FlexMatch.
Parâmetros só de visualização (`--cam_thr`, janela da oclusão) são argumentos de linha de comando e saem impressos na figura.

## Regras que o código impõe
- **test_public nunca entra na seleção.** Os sweeps nem o carregam (`run(..., evaluate_test=False)` recusa dados com test; há teste para isso). Cada run avalia o test **uma vez**, no checkpoint escolhido pelo val.
- Os diagnósticos de τ e de viés no test_public (Trilha B) são só reporte.
- `predict.py` carrega o checkpoint com `pretrained=False`, usa os nomes de classe e o pré-processamento guardados nele (`src/preproc.py`, o **mesmo** código do treino) e imprime `sum(p.numel() for p in model.parameters())`.

## Escolhas de implementação que vocês devem saber defender
- A vista forte do FixMatch parte do **mesmo recorte** da fraca (`UnlDS`): com animais de 2% do quadro, recortes independentes poderiam deixar o animal fora de uma das vistas.
- Aumentação `RandomGrayscale` (câmeras noturnas) e `resize_mode=pad` (proporções variam por câmera) são hipóteses; o sweep decide.
- Piso é re-tunado em cada orçamento; o FixMatch reaproveita a config do orçamento 5 nos orçamentos 1 e 20. Isso é conservador para o método e deve ser dito na apresentação.
- Com val de 202 imagens (2 de cavalo), diferenças menores que o desvio entre seeds são ruído; o sweep imprime média ± desvio.

## O que NÃO foi verificado (rode e confira)
- Rótulo nos CSVs como nome ou id numérico e o formato de `classes.csv` (o README do pacote só diz que tem “índice”): o `check_data` detecta e **falha alto** se ambíguo (`--image_col/--label_col` para forçar).
- Tempo real por passo em GPU e quantos trials cabem no Colab (`check_pipeline` mede).
- Se o FixMatch converge com 105 rótulos e animais minúsculos; se `resize_mode`/`size` importam; qualquer número de F1.
- Memória: o pool não rotulado em RAM é ~1,3 GB (size 224) a ~2,7 GB (320).

## Estrutura
```
configs/   project.json · defaults/ · sweeps/ · selected/ (gerado) · data_schema.json (gerado)
src/       diag_offline · comparar_ablacao · preproc · models · infer · common · part0 · fixmatch · sweep · run_final · select_final · saliencia · galeria · grafico_orcamento · avaliar
tests/     test_units · check_data · check_pipeline · check_predict · viz_aug · make_fake_data
colab/     PA3_colab.ipynb          scripts/  reproduzir.sh
predict.py · AI_LOG.md · pesos/ · resultados/ (runs/, sweeps/, parte2/, parte3/, curva_orcamento.csv)
```
Teste rápido sem os dados reais (CPU): `python -m tests.make_fake_data --out /tmp/fake && python -m tests.check_data --data /tmp/fake --fake`.
