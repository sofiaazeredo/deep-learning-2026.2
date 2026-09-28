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
| 0 — sintético, piso fácil | 5 elipses, speed 0,5, sem oclusão | 0,989 | 0 | 0 |
| 1 — baseline por quadro (teste 09+11) | SDP, Hungarian, IoU 0,2, min_hits=2, max_age=20 | 0,570 | 167 | 29,5 |
| 1 — as 7 sequências (figura) | mesma regra | 0,537 | 1674 | 78,0 |
| 2 — memória temporal (teste 09+11) | Trilha B, GRU, ResNet18 congelada, InfoNCE | 0,552 | 216 | 64,5 |
| 2 — as 7 sequências | mesma regra | 0,510 | 1930 | 170,4 |
| 3 — Eixo 3, teste 09+11 | aparência (melhor no teste), 3 seeds | 0,554 ± 0,001 | 215 ± 1 | 61,2 ± 4,0 |
| 3 — Eixo 3, as 7 | geometria (melhor no agregado), 3 seeds | 0,532 ± 0,013 | 1609 ± 164 | 137 ± 36 |
| 4 — correção (teste 09+11) | mesma GRU, IoU da última caixa depois do miss | 0,568 | 153 | 22,5 |
| 4 — as 7 | mesma regra | 0,546 | 1383 | 63,3 |
| 5 — detector leve (teste) | AP 0,582; temporal vs baseline | 0,464 vs 0,526 | 208 vs 189 | — |
| 5 — detector médio (teste) | AP 0,430 | 0,318 vs 0,347 | 351 vs 294 | — |
| 5 — detector forte (teste) | AP 0,197 | 0,170 vs 0,180 | 574 vs 510 | — |

O temporal **amplifica** a falha do detector: no leve o IDF1 cai 21% (0,568 → 0,464) enquanto o AP cai 16% e o baseline só 8%. Figura: `experiments/figures/stress_detector.png`.

Todo número aqui sai dos comandos da seção "Reproduzir cada parte".

**Decisões congeladas na Parte 1:**

- **fonte de detecções — SDP.** AP@0,5 / recall depois do nosso NMS, nas 7 cenas: DPM 0,356 / 0,365; FRCNN 0,541 / 0,543; SDP 0,653 / 0,655. O mesmo tracker (regra congelada) dá IDF1 0,310 (DPM), 0,487 (FRCNN), 0,537 (SDP). Só em 13 o FRCNN ganha um pouco em AP (0,582 vs 0,573); no resto o SDP é melhor. Congelado daqui em diante.
- **split por sequência — teste = 09 + 11; treino = 02, 04, 05, 10, 13.** Fixo, independente da seed. 09 é rua estática que o treino não viu; 11 é o único indoor (shopping, câmera móvel). O treino ainda tem as duas câmeras, o extremo de densidade (04) e o de oclusão (02 = 29, 13 = 3).
- **eixo da figura — duração de oclusão:** mediana das corridas com `visibility < 0,25`. Ordem: 13 (3) → 05 (5) → 10 (7) → 11 (9) → 09 (12) → 04 (20) → 02 (29). É o campo do gt, não buraco de quadro.
- **associação:** última caixa observada vs. detecção; Hungarian (guloso fica 0,532 nas 7, com 1906 switches contra 1674); limiar, `min_hits` e `max_age` varridos só no treino (melhor IDF1 0,523 em IoU 0,2 / min_hits=2 / max_age=20). `min_hits` é consecutivo; quadros antes da confirmação e quadros sem match não saem.
- **trilha da Parte 2 — B, RNN como memória de aparência.** O baseline quebra quando a pessoa some e reaparece longe da última caixa; um modelo de movimento (Trilha A) não alcança isso. O GRU agrega o embedding do recorte; o portão de IoU só vale no quadro em que a track acabou de ser vista.
- **eixo da Parte 3 — o que entra na recorrência.** Só geometria, só aparência, ou os dois. 3 seeds (42, 123, 7). No teste a aparência fica à frente (0,554 ± 0,001); nas sete a geometria ganha (0,532 ± 0,013) porque segura a cena densa 04. Os dois juntos não pegam o melhor de cada um.
- **correção da Parte 4 — `miss_prefer_iou`.** A GRU não esquece (o gradiente em k=14 ainda é ~90% do de k=0); o que mata a identidade é o casamento depois do miss. Preferir o IoU da última caixa nas tracks perdidas, e só então o cosseno, sobe o teste de 0,552 para 0,568 e as sete de 0,510 para 0,546 — acima do baseline (0,537).
- **estresse da Parte 5 — qualidade do detector.** Sem retreinar. O SDP
  público é descartado / ruidoso / injetado de FP em 3 intensidades
  (`leve` 10%/2%/5%, `média` 25%/5%/15%, `forte` 50%/10%/30%). O
  temporal (GRU + `miss_prefer_iou`) **amplifica** a falha: no leve o
  IDF1 do teste cai 21% e o do baseline 8%, com AP caindo 16%. Na 04
  densa o cosseno casa FP e recortes deslocados; o baseline que só
  olha IoU sofre menos. Em `forte` os dois colapsam junto com o AP
  (~0,17). Queda de taxa de quadros ficou de fora.

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
python scripts/train_temporal.py --track b --cell gru --window 16 --seed 42 --name modelo_final
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
python scripts/baseline_synthetic.py --sweep all --name synthetic_sweep

# Parte 1 — baseline por quadro
python scripts/prepare_mot17.py --measure
python scripts/run_baseline.py --detector SDP --sweep --name baseline
python scripts/run_baseline.py --detector SDP --matcher hungarian \
  --iou-threshold 0.2 --min-hits 2 --max-age 20 --name baseline
python scripts/evaluate_tracking.py --tracks experiments/results/baseline_tracks.csv \
  --detector SDP --name baseline --split all
python scripts/plot_baseline_results.py --order-by occlusion

# Parte 2 — memória temporal (a fonte de detecções fica congelada daqui em diante)
python scripts/train_temporal.py --track b --cell gru --window 16 --stride 8 --seed 42 --name temporal
python scripts/run_tracker.py --checkpoint checkpoints/temporal_best.pt --name temporal
python scripts/evaluate_tracking.py --tracks experiments/results/temporal_tracks.csv --name temporal

# Parte 3 — ablação, 3 seeds (Eixo 3: o que entra na recorrência)
for input in appearance geometry both; do
  for seed in 42 123 7; do
    python scripts/train_temporal.py --track b --input $input --cell gru \
      --window 16 --stride 8 --seed $seed --name input_${input}_seed${seed}
    python scripts/run_tracker.py --checkpoint checkpoints/input_${input}_seed${seed}_best.pt \
      --name input_${input}_seed${seed}
    python scripts/evaluate_tracking.py \
      --tracks experiments/results/input_${input}_seed${seed}_tracks.csv \
      --name input_${input}_seed${seed}
  done
done
python scripts/summarize_ablation.py --axis input

# Parte 4 — galeria de falhas, horizonte de memória e a correção
python scripts/failure_gallery.py --n-failures 3 --name failures
python scripts/memory_horizon.py --mode both --name memory_horizon
python scripts/failure_correction.py --name correcao

# Parte 5 — qualidade do detector (sem retreinar, modelo final + correção)
python scripts/stress_test.py --mode detector --name stress_detector
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
  test_detector_sim.py simulador de detector: descarte, ruído, falsos positivos
  test_metrics.py     os três casos à mão + AP@0.5 e drop de distractor
  test_dataset.py     parse MOT, pedestres vs distractores, split fixo
  test_detection.py   NMS contra resultado conhecido
  test_association.py custo IoU, cosseno, guloso e Hungarian
  test_tracker.py     persistência, min_hits, morte, miss_prefer_iou
  test_appearance.py  recorte preso à imagem, encoder L2 congelado
  test_model.py       passo do AppearanceRNN
  test_losses.py      InfoNCE: mesmo id mais perto que id diferente
  test_training.py    BPTT truncado e perfil ||∂L/∂h_{t-k}||
  test_baseline_synthetic.py  piso fácil IDF1 ≈ 1; oclusão longa dói
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
  casamento por quadro. ID switches e fragmentações usam o casamento por
  quadro do CLEAR MOT (definição do TrackEval): o par (gt, predição) do
  quadro imediatamente anterior continua se o IoU ainda passa do limiar, e só
  o resto vai para o Hungarian — sem isso,
  duas trajetórias que se cruzam contam switch à toa. Fragmentação só conta
  interrupção em quadro em que a identidade está no ground truth (oclusão
  total não fragmenta). Limiar de IoU 0,5, inclusivo.
- **Split.** Por sequência, nunca por quadro. Teste = 09 (estática, rua) e 11
  (móvel, indoor); treino = 02, 04, 05, 10, 13. A tabela da Parte 1 no topo
  usa o teste; a figura do descolamento usa as sete.
- **GT e distractores.** Só `conf=1, class=1` entra no ground truth. Como no
  TrackEval, as predições de cada quadro casam (Hungarian, IoU ≥ 0,5) com
  *todo* o gt do quadro, e só a que cai num distractor oficial — classes 2
  (pessoa em veículo), 7 (pessoa estática), 8 (distractor), 12 (reflexo) —
  sai antes da métrica. Predição em carro ou oclusor (3, 4, 5, 6, 9, 10, 11)
  fica e conta como FP. Validado contra o TrackEval nas execuções salvas:
  IDF1, switches, fragmentações, MOTA, FP e FN idênticos nas 7 cenas.
- **AP de detecção.** `average_precision` em IoU 0,5, ranqueada por score,
  implementação nossa. É o painel de cima da figura, não o IDF1.
- **Associação / nascimento / morte.** Compara com a última caixa observada
  (sem modelo de movimento). Hungarian default. Qualquer detecção sem par
  nasce id novo. A track só é emitida depois de `min_hits` matches
  consecutivos; os quadros anteriores são descartados (online). Sem match
  não se escreve a caixa velha. A track morre depois de `max_age` misses.
  Números (IoU 0,2, min_hits=2, max_age=20) vêm do sweep no treino; o
  default do stub (0,3 / 3 / 30) fica em 0,514 de IDF1 no mesmo treino.
- **NMS.** Greedy por score, IoU ≥ 0,5 suprime. `torchvision.ops.nms` não
  entra. Faster R-CNN do torchvision está implementado e grava `det.txt`;
  o laço de rastreamento só lê arquivo. Os números desta parte usam o
  `det.txt` público do SDP — os quadros do MOT17.zip ainda não foram
  extraídos.
- **Piso fácil (Parte 0.4).** Mesmo tracker da Parte 1, detecções = caixas
  verdadeiras sem id (`degrade` sem ruído). 5 elipses lentas, sem oclusão:
  IDF1 = 0,989 (os dois primeiros quadros de cada track somem por
  `min_hits=2`), zero switches, zero erro de contagem, 3 seeds. Densidade
  sozinha não quebra (15 objetos lentos continuam em 0,989). Velocidade
  quebra: 0,840 em 4 px/quadro e 0,447 em 8 — o deslocamento passa do
  portão de IoU. Oclusão de um alvo entre 3 objetos derruba o IDF1 para
  ~0,84 assim que há buraco; alongar o buraco quase não piora o global
  porque o gt some nesses quadros e as outras duas identidades pesam mais.
  Figura: `experiments/figures/synthetic_sweep.png`.
- **Trilha B (Parte 2).** Encoder ImageNet ResNet18 congelado, recorte
  128×64, projeção determinística 128-d L2. Cache em
  `data/MOT17/cache/sdp_resnet18/` (gt no treino, det nas 7 cenas). GRU
  hidden 128, teacher forcing, InfoNCE na janela T=16 (stride 8). Dois
  estágios: tracks recém-vistas casam por IoU (igual ao baseline);
  depois de um miss, 1 − cosseno entre a memória da GRU e a consulta
  (um passo a partir do estado zero), limiar 0,5. Sem isso o
  reaparecimento longe da caixa velha continua bloqueado — e casar
  aparência em todo quadro derruba o IDF1 para ~0,12. No teste fica
  abaixo do baseline (0,552 vs 0,570): ganha em 10 e 13 (câmera
  móvel), perde nas cenas densas 02 e 04, onde o ResNet confunde
  pedestres parecidos. Kalman e Trilha A ficam de fora.
- **Eixo 3 (Parte 3).** Mesmo GRU, T=16, teacher forcing. Caixa em
  `(cx, cy, w, h)` normalizado pela imagem. Depois do miss: aparência =
  cosseno; geometria = IoU da caixa *prevista* (a track roda para
  frente sem emitir caixa); os dois = `min(cosseno, 1 − IoU prevista)`,
  limiar 0,5. 3 seeds. No teste (09+11, densidade ~10) a aparência
  ganha por pouco e com menor desvio (0,554 ± 0,001 vs geometria
  0,552 ± 0,009). Nas sete, a geometria ganha (0,532 ± 0,013) — o
  salto está em 04 (0,659 vs 0,607), a cena mais densa. Em 02 (oclusão
  29) os três caem juntos e nenhum recupera o baseline (0,414). Fundir
  as duas cabeças não herda o melhor dos dois. A pergunta do enunciado:
  quem sustenta a identidade numa oclusão longa *muda com a densidade*
  — geometria na multidão, aparência no teste mais espaçado.
- **Galeria e horizonte (Parte 4).** Três falhas do checkpoint de
  aparência seed 42: miss de 1 quadro na 04 densa (ids vizinhos
  trocados), oclusão de 24 quadros na 02 (acima de `max_age=20`, a
  track morre), buraco de 16 quadros na 09 de teste (cabe exatamente
  em T). Analítico: `||∂L_t/∂h_{t-k}||` da InfoNCE no último quadro
  com positivo no passado; a GRU segura ~90% do gradiente até k=14 —
  não é o gradiente que some. Empírico: 118 de 1111 oclusões
  (`visibility < 0,25`) voltam com o mesmo id; a mediana das que
  sobrevivem é 3 quadros, a do dataset é 8. A correção
  `miss_prefer_iou` ataca o diagnóstico da 04: nas tracks perdidas o
  Hungarian de IoU da última caixa roda antes do cosseno. Teste
  0,552 → 0,568 (quase o baseline 0,570); as sete 0,510 → 0,546 e
  1930 → 1383 switches, passando o baseline. O ganho grande é 02 e
  04; 10 e 13 mal andam — oclusão longa + câmera móvel não se resolve
  com a caixa velha.
- **Estresse do detector (Parte 5).** Sem retreinar, em cima da GRU
  seed 42 com `miss_prefer_iou`. O `degrade` da Parte 0.2 agora
  também aceita o `det.txt` (`degrade_detections`): mesma semente por
  cena, então o descarte é aninhado (forte ⊂ média ⊂ leve). mAP e
  IDF1 no mesmo gráfico. No SDP limpo o temporal ainda ganha nas sete
  (0,546 vs 0,537). Qualquer sujeira inverte: o cosseno trata um
  recorte ruidoso ou um FP como reaparecimento, e o IDF1 cai mais que
  o AP e mais que o baseline. A memória de aparência assume que a
  caixa que chegou é uma pessoa; o detector que a inventou quebra
  essa premissa.
- **MOTA.** Opcional e reportada à parte: com o detector congelado, os termos de
  FP/FN quase não variam entre as configurações e ela esconde o que muda.
