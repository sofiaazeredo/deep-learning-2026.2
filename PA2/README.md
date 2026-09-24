# PA2 — Identidade ao longo do tempo

Aprendizado Profundo (FGV) — Programming Assignment 2.

Fazer as arquiteturas das aulas de detecção e de RNN produzirem rótulos
**identity-aware**: se um objeto aparece no quadro 3 e reaparece no quadro 40,
ele sai com o mesmo identificador. Sem rastreador pronto (ByteTrack, DeepSORT,
OC-SORT, `model.track` do ultralytics e afins), sem métrica de rastreamento de
biblioteca (motmetrics, TrackEval) e sem `torchvision.ops.nms` — IDF1, ID
switches, fragmentações, associação, gestão de tracks e NMS são implementação
nossa.

No PA1 fomos de *class-aware* para *instance-aware* no espaço; aqui vamos de
*instance-aware por quadro* para *identity-aware no tempo*.

Dataset: **MOT17 / MOTChallenge**, pedestres em rua e ambiente interno, com
caixas e identidades anotadas quadro a quadro.

---

## Resultados

| Parte | Configuração | IDF1 | ID switches | Erro de contagem |
|---|---|---|---|---|
| 0 — sintético, piso fácil | associação ingênua | — | — | — |
| 1 — baseline por quadro | IoU t vs. t−1 | — | — | — |
| 2 — memória temporal | (trilha a definir) | — | — | — |

(preencher conforme as execuções saírem; todo número aqui tem que ser
reproduzível pelos comandos da seção "Reproduzir cada parte".)

**Decisões a registrar aqui**, porque o enunciado cobra a justificativa:

- **fonte de detecções** — qual dos três detectores públicos (DPM, FRCNN, SDP)
  é a fonte padrão do resto do PA, e por quê;
- **split por sequência** — qual sequência inteira ficou fora e sob que
  critério (câmera parada vs. móvel, densidade, ponto de vista);
- **trilha da Parte 2** — A (RNN como modelo de movimento) ou B (RNN como
  memória de aparência);
- **eixo da Parte 3** — célula recorrente, regime de treino, o que entra na
  recorrência, ou direção do contexto;
- **estresse da Parte 5** — queda de taxa de quadros ou qualidade do detector;
- **regra de associação e gestão de nascimento/morte de tracks** — limiar de
  IoU, `max_age`, `min_hits`, guloso ou Hungarian. Como no PA1, regras
  diferentes dão números diferentes.

---

## Ambiente

Python 3.11+ com PyTorch. No Colab não é preciso instalar nada além do que já
vem.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

Os scripts inserem a raiz do projeto no `sys.path` sozinhos, então rodam de
qualquer diretório sem `PYTHONPATH`.

## Dados

Baixe o MOT17 de <https://motchallenge.net/data/MOT17/> (download direto, sem
conta) e coloque em `PA2/data/MOT17/`:

```
data/MOT17/train/MOT17-02-SDP/
    seqinfo.ini
    gt/gt.txt        # frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    det/det.txt      # detecções públicas do detector da pasta
    img1/000001.jpg  # ...
```

O pacote completo tem ~5,5 GB. **Comece pelo pacote só de anotações (~10 MB)**:
dá para escrever e depurar a métrica e a associação inteiras — Parte 0 e boa
parte da Parte 1 — antes de baixar um único quadro.

Os dados **não** estão versionados (ver `.gitignore`).

```bash
python scripts/prepare_mot17.py --root data/MOT17   # confere o layout e imprime o split
```

## Um comando que treina

```bash
python scripts/train_temporal.py --track a --cell gru --window 16 --seed 42 --name modelo_final
```

## Um comando que avalia

```bash
python scripts/run_tracker.py --checkpoint checkpoints/modelo_final_best.pt --name modelo_final
python scripts/evaluate_tracking.py --tracks experiments/results/modelo_final_tracks.csv --name modelo_final
```

## Inferência numa sequência qualquer

Sem retreinar:

```python
from src.inference import track_sequence

result = track_sequence("data/MOT17/train/MOT17-02-SDP")
print(result["count"])     # objetos únicos no vídeo
result["frames"]           # quadros com as identidades coloridas
```

O notebook `inferencia.ipynb` faz o mesmo e escreve o vídeo.

---

## Testes

Todos os testes ficam em `tests/`, um arquivo por módulo de `src/`
(`tests/test_synthetic.py` testa `src/synthetic.py`, e assim por diante). Cada
peça nova entra com o teste junto, e a suíte inteira roda a cada passo:

```bash
python -m pytest tests/                  # a suíte inteira
python -m pytest tests/test_synthetic.py # um módulo
python tests/test_synthetic.py           # o mesmo, sem pytest
```

Enquanto uma parte não está implementada, o teste dela falha com
`NotImplementedError` — é o sinal de que falta aquela parte, não de que algo
quebrou.

---

## Reproduzir cada parte

O split é por sequência e fixo, independente da seed de treino, então todas as
execuções compartilham exatamente o mesmo conjunto de teste.

```bash
# Parte 0 — sintético (não precisa do MOT17)
python scripts/generate_synthetic.py --name synthetic
python tests/test_metrics.py                         # os três casos à mão
python scripts/baseline_synthetic.py --sweep occlusion --name synthetic_sweep

# Parte 1 — baseline por quadro
python scripts/run_baseline.py --detector SDP --matcher hungarian --name baseline
python scripts/evaluate_tracking.py --tracks experiments/results/baseline_tracks.csv --name baseline
python scripts/plot_baseline_results.py --order-by density

# Baseline permitido de comparação (Kalman de velocidade constante)
python scripts/compare_kalman.py --name kalman

# Parte 2 — memória temporal (a fonte de detecções fica congelada daqui em diante)
python scripts/train_temporal.py --track a --cell gru --window 16 --seed 42 --name temporal
python scripts/run_tracker.py --checkpoint checkpoints/temporal_best.pt --name temporal
python scripts/evaluate_tracking.py --tracks experiments/results/temporal_tracks.csv --name temporal

# Parte 3 — ablação, 3 seeds (exemplo com o Eixo 1: a célula recorrente)
for cell in rnn lstm gru; do
  for seed in 42 123 7; do
    python scripts/train_temporal.py --cell $cell --window 16 --seed $seed \
      --name cell_${cell}_seed${seed}
    python scripts/run_tracker.py --checkpoint checkpoints/cell_${cell}_seed${seed}_best.pt \
      --name cell_${cell}_seed${seed}
    python scripts/evaluate_tracking.py \
      --tracks experiments/results/cell_${cell}_seed${seed}_tracks.csv \
      --name cell_${cell}_seed${seed}
  done
done
python scripts/summarize_ablation.py --axis cell

# Parte 4 — galeria de falhas, horizonte de memória e a correção
python scripts/failure_gallery.py --n-failures 3 --name failures
python scripts/memory_horizon.py --mode both --name memory_horizon
python scripts/failure_correction.py --name correcao

# Parte 5 — teste de estresse (escolher UM)
python scripts/stress_test.py --mode framerate
python scripts/stress_test.py --mode detector
```

A partir da Parte 2 os scripts leem `experiments/results/best_model.json` e
resolvem o checkpoint sozinhos; `--checkpoint` força outro.

## Checkpoint

Os pesos do modelo temporal ficam em `checkpoints/`. Se passarem do limite de
arquivo do GitHub (100 MB), entram por link do Drive, como no PA1:

**(link aqui)**

---

## Estrutura

```
src/
  synthetic.py        gerador de vídeos de elipses com oclusão real (Parte 0)
  detector_sim.py     degradação proposital das detecções (Partes 0 e 5)
  metrics.py          IDF1, ID switches, fragmentações, erro de contagem
  dataset.py          MOT17, split por sequência, janelas de T quadros
  detection.py        detecções públicas + torchvision, NMS próprio
  association.py      custos (IoU, cosseno), guloso/Hungarian, portão
  tracker.py          estado da track, nascimento/morte, laço de rastreamento
  model.py            RNN/LSTM/GRU do modelo temporal (trilhas A e B)
  appearance.py       recorte -> embedding, encoder congelado (trilha B)
  losses.py           smooth-L1, NLL gaussiana, triplet, contrastiva
  training.py         BPTT truncado, regimes de treino, perfil de gradiente
  inference.py        sequência qualquer -> vídeo com IDs + contagem
scripts/              um script por etapa (ver acima)
tests/                testes de cada módulo de src/, rodados com pytest
  test_synthetic.py   gerador: formato, determinismo, oclusão de N quadros
  test_metrics.py     os três casos à mão da métrica (Parte 0.3)
experiments/
  results/            um CSV por execução + os sumários das ablações
  figures/            todas as figuras da apresentação
inferencia.ipynb      entregável de inferência, roda sem retreinar
AI_LOG.md             uso de IA neste trabalho
```

## Notas de método

(preencher conforme as decisões forem tomadas — as mesmas categorias do PA1:
métrica, split, resolução/escala, e o que ficou fora.)

- **Métrica de identidade.** IDF1 exige uma atribuição global um-para-um entre
  identidades previstas e verdadeiras ao longo da sequência inteira, não um
  casamento por quadro. O matching por IoU reaproveita o do PA1, como o
  enunciado permite.
- **Split.** Por sequência, nunca por quadro: separar quadros aleatoriamente
  põe o quadro t no treino e o t+1 na validação, e o modelo temporal seria
  avaliado em cima do que já viu.
- **MOTA.** Opcional e reportada à parte: com o detector congelado, os termos de
  FP/FN quase não variam entre as configurações e ela esconde o que muda.
