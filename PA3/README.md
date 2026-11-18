# PA3 — Poucos rótulos (fauna em câmeras trap) · Trilha B (FixMatch)

Classificação de 21 espécies (ENA24-detection) com 5 rótulos por classe e as 6.505 imagens do `train` sem rótulo.
Modelo entregue: EfficientNet-B0 (pesos ImageNet, fine-tuning), 4.034.449 parâmetros (limite: 12M), inferência em CPU.

## Resultados (F1 macro no `test_public`)

| Rótulos por classe | Piso (ImageNet, só rótulos) | FixMatch | Teto |
|---|---|---|---|
| 1 | 0,223 | 0,248 | |
| 5 (média ± desvio, 3 seeds) | 0,463 ± 0,030 | 0,488 ± 0,014 | 0,686 |
| 20 | 0,625 | 0,668 | |

Parte 0 (orçamento 5, seed 0): CNN do zero F1 0,296 e acurácia 0,319; backbone ImageNet F1 0,428 e acurácia 0,501.
Os números por run estão em `resultados/curva_orcamento.csv` e `resultados/parte2/tabela_parte2.csv`.
O modelo entregue é o run `fixmatch_b5_s2` (F1 0,483, acurácia 0,530 no `test_public`), escolhido pelo F1 no val.

## Ambiente
Python ≥ 3.10, PyTorch e torchvision; `pip install -r requirements.txt`. Os experimentos foram executados no Colab (GPU T4).
Os pesos ImageNet são baixados pelo torchvision apenas durante o treino; o `predict.py` não acessa a rede.

## Dados
Pacote do enunciado (`pa3_dados.zip`) descompactado em `dados/` (no Colab, `/content/dados`), com o layout:
`images/{train,val,test_public}/*.jpg`, `labels/{rotulos_1,rotulos_5,rotulos_20,val,test_public}.csv` (colunas `image_id,label`; `image_id` é o nome do arquivo sem `.jpg`) e `classes.csv`.
Não há `train.csv`: o conjunto não rotulado é a pasta `images/train/`. O comando `python -m tests.check_data` confere o pacote contra o enunciado e grava o schema detectado em `configs/data_schema.json`.

Origem: ENA24-detection (LILA BC, CDLA-Permissive-1.0). Citação pedida pelos autores: Yousif H, Kays R, He Z. *Dynamic Programming Selection of Object Proposals for Sequence-Level Animal Species Classification in the Wild.* IEEE TCSVT, 2019.

## Um comando que treina
```bash
bash scripts/reproduzir.sh dados resultados
```
Executa os testes, a verificação dos dados, os sweeps de hiperparâmetros (somente no val), a Parte 0, a Parte 2, a escolha do modelo entregue, a galeria da Parte 3 e o teste do `predict.py`.
Tudo é retomável: sweeps e runs concluídos são pulados. O tamanho dos sweeps e o lote do FixMatch são variáveis de ambiente (`N_FT`, `N_SCRATCH`, `N_FM`, `FM_STEPS_SWEEP`, `FM_BS`, `SWEEP_SEEDS`, `CACHE`).

O notebook `colab/PA3_colab.ipynb` executa a mesma sequência, célula a célula, e inclui também os diagnósticos da Trilha B (seção 7: calibração, τ com confiança bruta, ablação sem correção de desbalanceamento, análise com 3000 passos).

## Um comando que avalia
```bash
python predict.py --imagens PASTA --saida preds.csv
python -m src.avaliar --preds preds.csv --gt dados/labels/test_public.csv --data dados
```
O `predict.py` escreve um CSV com colunas `image_id,label` (nomes de `classes.csv`), roda em CPU, não usa internet e imprime o número de parâmetros do modelo.
Os pesos ficam em `pesos/modelo_final.pt`. O `src.avaliar` imprime F1 macro e acurácia.

## Fluxo
| Etapa | Comando | Produz |
|---|---|---|
| Testes de código | `python -m tests.test_units` | 12 testes (pré-processamento, FlexMatch, alinhamento, EMA, Grad-CAM, checkpoint) |
| Dados | `python -m tests.check_data --data dados` | conferência do pacote e `configs/data_schema.json` |
| Pipeline | `python -m tests.check_pipeline --data dados` | limite de parâmetros, overfit sem aumentação, seed, guarda do `test_public`, FixMatch nas 4 correções, tempo por passo |
| Aumentações | `python -m tests.viz_aug --data dados` | figura das aumentações |
| Hiperparâmetros | `python -m src.sweep --method {part0_scratch,part0_ft,fixmatch}` | busca aleatória no val → `configs/selected/*.json` |
| Parte 0 | `python -m src.part0 --data dados` | piso (a) CNN do zero e (b) backbone ImageNet; curva, matriz de confusão, época em que decora |
| Partes 1 (B) e 2 | `python -m src.run_final --data dados` | piso e FixMatch nos orçamentos 1/5/20 (3 seeds no 5) → `resultados/curva_orcamento.csv`, diagnósticos de τ e de viés |
| Gráficos | `python -m src.grafico_orcamento` | curva de orçamento (x em log, teto 0,686) e ganho por classe |
| Diagnósticos offline | `python -m src.diag_offline --data dados --runs resultados/runs/fixmatch_b5_s0 ...` | ECE, confiabilidade, τ bruto, erro em imagens cinza × coloridas |
| Ablação | `python -m src.comparar_ablacao resultados resultados_sem_correcao ...` | val e teste F1 (média ± desvio) e diferença pareada por seed |
| Modelo entregue | `python -m src.select_final --data dados` | escolhe a seed pelo val → `pesos/modelo_final.pt` |
| Parte 3 | `python -m src.galeria --data dados` | 3 erros e 5 acertos com Grad-CAM e oclusão |
| Entrega | `python -m tests.check_predict --data dados` | CPU, rede bloqueada, formato do CSV, tempo |

## Onde cada decisão é tomada
`configs/defaults/*.json` são os pontos de partida dos sweeps. Os resultados usam apenas `configs/selected/*.json`, gerados por `src/sweep.py` com escolha no val. Sem esses arquivos os scripts recusam rodar, a menos que se passe `--allow_defaults` (o que fica registrado em `metrics.json`).

| Decisão | Espaço de busca (`configs/sweeps/`) | Critério |
|---|---|---|
| Backbone (resnet18, mobilenet_v3_large, efficientnet_b0), `data.size`, `data.resize_mode` | `part0_ft.json` | F1 macro no val do piso, orçamento 5; o FixMatch herda a escolha |
| lr, wd, épocas, escala do recorte | `part0_*.json` | F1 médio no val entre seeds (desempate: val loss) |
| τ, correção (`none`/`da`/`flex`/`both`), λ, lr, escala do recorte, magnitude do RandAugment | `fixmatch.json` | F1 médio no val |
| Época ou passo do checkpoint | — | F1 no val dentro de cada run |
| Modelo entregue | — | maior F1 no val entre as 3 seeds do orçamento 5 |

Configuração selecionada do FixMatch (orçamento 5): EfficientNet-B0, entrada 320 com redimensionamento por squash, 1.500 passos, lote rotulado 16, μ = 3, lr 0,00684, τ = 0,95, correção `flex` (limiar por classe com piso 0,5), λ = 1.
A vista forte do FixMatch parte do mesmo recorte da vista fraca. O piso é re-tunado em cada orçamento; o FixMatch reaproveita a configuração do orçamento 5 nos orçamentos 1 e 20.

Constantes do enunciado ficam em `configs/project.json` (21 classes, limite de 12M parâmetros, teto 0,686, tamanhos dos conjuntos).
Constantes de algoritmo do FixMatch e do FlexMatch (cosseno `cos(7πk/16K)`, função `x/(2−x)`) não são ajustadas.

## Regras impostas pelo código
- O `test_public` não entra na seleção: os sweeps nem o carregam (`run(..., evaluate_test=False)` recusa dados com teste; há teste unitário). Cada run avalia o teste uma única vez, no checkpoint escolhido pelo val.
- Os diagnósticos de τ e de viés da Trilha B no `test_public` são apenas reporte.
- O `predict.py` carrega o checkpoint com `pretrained=False`, usa os nomes de classe e o pré-processamento guardados nele (`src/preproc.py`, o mesmo código do treino) e imprime `sum(p.numel() for p in model.parameters())`.
- Nenhum rótulo manual, API de modelo fechado ou dado externo é usado.

## Estrutura
```
configs/   project.json · defaults/ · sweeps/ · selected/ · data_schema.json (gerado)
src/       preproc · models · infer · common · part0 · fixmatch · sweep · run_final · select_final · saliencia · galeria · grafico_orcamento · diag_offline · comparar_ablacao · avaliar
tests/     test_units · check_data · check_pipeline · check_predict · probe_memory · viz_aug · make_fake_data
colab/     PA3_colab.ipynb          scripts/  reproduzir.sh
predict.py · AI_LOG.md · pesos/ · resultados/ (runs/, sweeps/, parte2/, parte3/, diag_offline/, curva_orcamento.csv)
```
Teste rápido sem os dados reais (CPU): `python -m tests.make_fake_data --out /tmp/fake --real_layout && python -m tests.check_data --data /tmp/fake --fake`.
