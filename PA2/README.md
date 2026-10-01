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

IDF1 é a média por sequência (cada cena pesa igual); switches somam; erro de
contagem é a média de |ids previstos − ids verdadeiros|. ± é o desvio
amostral entre 3 seeds. Teste = 09 + 11; "as 7" inclui o treino.

| Parte | Configuração | IDF1 | ID switches | Erro de contagem |
|---|---|---|---|---|
| 0 — sintético, piso fácil | 5 elipses, speed 0,5, sem oclusão, 3 seeds | 0,989 | 0 | 0 |
| 1 — baseline por quadro (teste) | SDP, Hungarian, IoU 0,2, min_hits=2, max_age=20 | 0,570 | 176 | 29,5 |
| 1 — as 7 (figura) | mesma regra | 0,536 | 1787 | 78,3 |
| 2 — Trilha B (teste) | GRU, ResNet18 congelada, perda preditiva, cosseno com portão | 0,576 | 155 | 29,5 |
| 2 — as 7 | mesma regra | 0,552 | 1399 | 75,7 |
| 2 — controle (teste / 7) | mesmo tracker, cosseno desligado | 0,575 / 0,550 | 155 / 1408 | 32,0 / 82,3 |
| 3 — Eixo 3, teste | geometria (melhor), 3 seeds | 0,585 ± 0,002 | 151 ± 2 | 26,7 ± 0,3 |
| 3 — Eixo 3, as 7 | geometria (melhor), 3 seeds | 0,558 ± 0,001 | 1394 ± 7 | 67,4 ± 0,8 |
| 4 — correção (teste) | Trilha B + `prefer_confirmed` | 0,578 | 134 | 21,0 |
| 4 — correção (as 7) | mesma | 0,565 | 1200 | 53,9 |
| 5 — detector leve (teste) | AP 0,599; modelo final vs baseline | 0,483 vs 0,526 | 180 vs 194 | — |
| 5 — detector médio (teste) | AP 0,444 | 0,376 vs 0,346 | 281 vs 300 | — |
| 5 — detector forte (teste) | AP 0,208 | 0,182 vs 0,180 | 390 vs 527 | — |

O ganho da Trilha B sobre o baseline vem quase todo da cascata de
associação (IoU da última caixa nas tracks perdidas): o controle com o
cosseno desligado fica em 0,575 / 0,550. A memória recorrente soma +0,001
no teste e +0,002 no treino — ver "Notas de método".

A avaliação segue o protocolo do TrackEval (distractores, continuidade do
CLEAR MOT) e foi conferida contra ele: ver "Notas de método".

Todo número aqui sai dos comandos da seção "Reproduzir cada parte".

**Decisões congeladas** — toda escolha foi feita olhando só o treino
(02, 04, 05, 10, 13); o teste só é lido depois:

- **fonte de detecções — SDP.** AP@0,5 / recall médio nas cenas de treino
  (ranking inteiro, depois do nosso NMS): DPM 0,349 / 0,376; FRCNN
  0,523 / 0,526; SDP 0,633 / 0,640; Faster R-CNN do torchvision
  0,608 / 0,690. IDF1 no treino com a regra congelada: SDP 0,522, FRCNN
  0,463, torchvision 0,461, DPM 0,269. O torchvision (segunda fonte, score ≥ 0,8 escolhido por sweep
  no treino) tem o maior recall e quase o AP do SDP, mas cria identidades
  demais: detector melhor não é rastreamento melhor.
- **split — teste = 09 + 11; treino = 02, 04, 05, 10, 13.** Por sequência,
  fixo, independente da seed. O teste tem uma câmera parada (09) e uma
  móvel (11); o movimento do teste fica dentro do que o treino cobre (a 13,
  a mais rápida, fica no treino).
- **eixo da figura — duração de oclusão:** mediana das corridas com
  `visibility < 0,25`. Ordem: 13 (3) → 05 (5) → 10 (7) → 11 (9) → 09 (12)
  → 04 (20) → 02 (29). É o campo do gt, não buraco de quadro.
- **associação (Parte 1):** última caixa observada vs. detecção; Hungarian
  (guloso: 0,517 no treino vs 0,522); limiar, `min_hits` e `max_age` do
  sweep de 60 configurações no treino (melhor: IoU 0,2 / min_hits=2 /
  max_age=20, IDF1 0,522). `min_hits` é consecutivo; quadros antes da
  confirmação e quadros sem match não saem.
- **trilha da Parte 2 — B, RNN como memória de aparência.** Encoder
  ImageNet ResNet18 congelado; GRU agrega o embedding de cada observação.
  Perda preditiva: a memória depois do quadro t tem que reconhecer, entre
  os recortes das outras pessoas, os recortes FUTUROS da mesma pessoa
  (cabeça de consulta própria). Associação em cascata: IoU nas tracks vistas
  no quadro anterior → IoU da última caixa nas perdidas → cosseno memória ×
  consulta **com o portão geométrico** do enunciado (a última caixa cresce
  0,6 por quadro perdido; IoU ≥ 0,1; limiar 0,5 — sweep no treino).
- **eixo da Parte 3 — o que entra na recorrência** (geometria / aparência /
  os dois), 3 seeds. A geometria prevê o deslocamento da caixa na
  codificação do Faster R-CNN, com a velocidade na entrada.
- **correção da Parte 4 — `prefer_confirmed`.** Do diagnóstico medido nos
  switches do treino (40% são "não recasou", com buraco mediano de 3
  quadros): as tracks tentativas vivas só casam por IoU depois das
  confirmadas.
- **estresse da Parte 5 — qualidade do detector.** Sem retreinar, no modelo
  final (Trilha B + correção). SDP descartado / ruidoso / com FP em 3
  intensidades (`leve` 10%/2%/5%, `média` 25%/5%/15%, `forte` 50%/10%/30%).
  Resposta: **amplifica.** No treino o modelo final perde mais IDF1 que o
  baseline nas três intensidades (IDF1 mantido ×0,78 / 0,58 / 0,29 contra
  ×0,86 / 0,61 / 0,32 do baseline; o AP fica em ×0,85 / 0,62 / 0,28). No
  teste (2 cenas) o `médio` é a exceção (×0,65 contra ×0,61, puxado pela
  11); `leve` amplifica (×0,84 vs ×0,92) e no `forte` os dois caem junto
  com o AP (~0,18). O que se mantém nos dois splits e nas três
  intensidades: o modelo troca menos ids (treino, `forte`: 3607 → 2891).

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

Dois caches são gerados em `data/MOT17/cache/` e precisam dos **quadros**
(o pacote completo): `torchvision/` (detecções do Faster R-CNN,
`scripts/detect_torchvision.py`) e `sdp_resnet18/` (embeddings dos recortes
da Trilha B, criados pelo `train_temporal.py` / `run_tracker.py` na primeira
execução).

## Um comando que treina

```bash
python scripts/train_temporal.py --track b --cell gru --window 16 --seed 42 --name modelo_final
```

## Um comando que avalia

```bash
python scripts/run_tracker.py --checkpoint checkpoints/modelo_final_best.pt --prefer-confirmed --name modelo_final
python scripts/evaluate_tracking.py --tracks experiments/results/modelo_final_tracks.csv --name modelo_final
```

O modelo final é a Trilha B com a correção da Parte 4 (`--prefer-confirmed`);
sem a flag, é o tracker das Partes 2 e 3.

## Inferência numa sequência qualquer

Sem retreinar:

```python
from src.inference import track_sequence

result = track_sequence("data/MOT17/train/MOT17-02-SDP")
print(result["count"])     # objetos únicos no vídeo
result["frames"]           # quadros com as identidades coloridas
```

Qualquer pasta com quadros serve (`img1/` ou as imagens direto na pasta;
`seqinfo.ini` é opcional). Com `det/det.txt` as detecções saem dele; sem ele
roda o Faster R-CNN do torchvision. `checkpoint=False` roda o baseline da
Parte 1. O notebook `inferencia.ipynb` faz o mesmo e escreve o vídeo.

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

# Parte 1 — as duas fontes de detecção e o baseline por quadro
python scripts/detect_torchvision.py                 # Faster R-CNN do torchvision -> cache (~12 min, RTX 3050)
python scripts/prepare_mot17.py --measure            # AP/recall de DPM, FRCNN, SDP e torchvision
python scripts/run_baseline.py --detector SDP --sweep --name baseline               # a regra, só no treino
python scripts/run_baseline.py --detector torchvision \
  --sweep-score 0.3 0.5 0.6 0.7 0.8 0.9 --name baseline_torchvision                # limiar de score, só no treino
for det in SDP DPM FRCNN torchvision; do
  name=baseline; [ $det != SDP ] && name=baseline_$(echo $det | tr A-Z a-z)
  python scripts/run_baseline.py --detector $det --name $name
  python scripts/evaluate_tracking.py --tracks experiments/results/${name}_tracks.csv \
    --detector $det --name $name --split all
done
python scripts/run_baseline.py --detector SDP --matcher greedy --name baseline_greedy
python scripts/evaluate_tracking.py --tracks experiments/results/baseline_greedy_tracks.csv \
  --name baseline_greedy --split all
python scripts/evaluate_tracking.py --tracks experiments/results/baseline_tracks.csv \
  --name baseline_test --split test
python scripts/plot_baseline_results.py --order-by occlusion

# Parte 2 — Trilha B (a fonte de detecções fica congelada daqui em diante)
python scripts/train_temporal.py --track b --cell gru --window 16 --stride 8 --seed 42 --name temporal
python scripts/sweep_tracker.py                      # limiar do cosseno e portão geométrico, só no treino
python scripts/run_tracker.py --name temporal
python scripts/run_tracker.py --appearance-threshold -1 --name null_after_miss     # controle sem aparência
for name in temporal null_after_miss; do
  python scripts/evaluate_tracking.py --tracks experiments/results/${name}_tracks.csv --name $name
done
python scripts/evaluate_tracking.py --tracks experiments/results/temporal_tracks.csv \
  --name temporal_test --split test
python scripts/plot_temporal_vs_baseline.py          # lado a lado com o baseline + quadros da 09

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
python scripts/summarize_ablation.py --axis input    # IDF1 ± desvio e sobrevivência através da oclusão

# Parte 4 — galeria de falhas, horizonte de memória e a correção
python scripts/failure_gallery.py --n-failures 3 --name failures
python scripts/memory_horizon.py --mode both --name memory_horizon
python scripts/failure_correction.py --name correcao

# Parte 5 — qualidade do detector (sem retreinar, modelo final = Trilha B + correção)
python scripts/stress_test.py --mode detector --name stress_detector
```

A partir da Parte 2 os scripts leem `experiments/results/best_model.json` e
resolvem o checkpoint sozinhos; `--checkpoint` força outro.

## Checkpoint

O modelo final entra no repositório: `checkpoints/temporal_best.pt`
(~0,8 MB, Trilha B, GRU, perda preditiva, seed 42).
`experiments/results/best_model.json` aponta para ele com caminho relativo
ao PA2, então roda em qualquer máquina. Os checkpoints das ablações
(3 entradas × 3 seeds) não entram — `scripts/train_temporal.py` os
regenera.

---

## Estrutura

```
src/
  boxes.py            caixas normalizadas e a codificação de deslocamento (Faster R-CNN)
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
  test_tracker.py     persistência, min_hits, morte, cascata, portão, prefer_confirmed
  test_appearance.py  recorte preso à imagem, encoder L2 congelado
  test_model.py       passo do AppearanceRNN
  test_losses.py      InfoNCE: mesmo id mais perto que id diferente
  test_training.py    BPTT truncado e perfil ||∂L/∂h_{t-k}||
  test_baseline_synthetic.py  piso fácil IDF1 ≈ 1; oclusão longa dói
  test_boxes.py       codificação do deslocamento e velocidade por quadro
  test_inference.py   inferência numa pasta qualquer, sem det.txt, vídeo
  test_failure_gallery.py      tipos de switch da galeria (morreu / não recasou / troca)
  test_summarize_ablation.py   sobrevivência através da oclusão, por split
experiments/
  results/            um CSV por execução + os sumários das ablações
  figures/            todas as figuras da apresentação
inferencia.ipynb      entregável de inferência, roda sem retreinar
AI_LOG.md             uso de IA neste trabalho
```

## Notas de método

- **Métrica de identidade.** IDF1 exige uma atribuição global um-para-um
  entre identidades previstas e verdadeiras ao longo da sequência inteira,
  não um casamento por quadro. ID switches e fragmentações usam o casamento
  por quadro do CLEAR MOT (definição do TrackEval): o par (gt, predição) do
  quadro imediatamente anterior continua se o IoU ainda passa do limiar, e
  só o resto vai para o Hungarian — sem isso, duas trajetórias que se cruzam
  contam switch à toa. Limiar de IoU 0,5, inclusivo.
- **GT e distractores.** Só `conf=1, class=1` entra no ground truth. Como no
  TrackEval, as predições de cada quadro casam (Hungarian, IoU ≥ 0,5) com
  *todo* o gt do quadro, e só a que cai num distractor oficial — classes 2
  (pessoa em veículo), 7 (pessoa estática), 8 (distractor), 12 (reflexo) —
  sai antes da métrica. Predição em carro ou oclusor fica e conta como FP.
- **Validação da métrica.** Uma vez, fora do repositório e do ambiente do
  projeto, as trajetórias salvas passaram pelo TrackEval e pelo motmetrics
  só como gabarito: IDF1, switches, fragmentações e MOTA idênticos ao
  TrackEval nas 7 cenas em todas as execuções, com uma exceção conhecida —
  num quadro em que o tracker não emite nenhuma caixa, o TrackEval não zera
  a memória do quadro anterior e deixa de contar a fragmentação de quem
  atravessa o quadro vazio. Só aparece nas execuções com quadros vazios
  atravessados por alguém — torchvision (09, 11, 13) e DPM na 13; na 09 do
  torchvision a diferença é 5 = exatamente as identidades casadas dos dois
  lados do quadro 166. A nossa contagem é a
  do motmetrics, com quem bate exatamente. O NMS bate com
  `torchvision.ops.nms` e o AP com uma implementação VOC de referência.
  Nenhuma dessas bibliotecas é usada pelo código (ver AI_LOG).
- **AP de detecção.** `average_precision` em IoU 0,5, VOC com todos os
  pontos, no ranking inteiro de cada fonte (sem limiar de score). É o painel
  de cima da figura da Parte 1, não o IDF1.
- **NMS.** Guloso por score, IoU ≥ 0,5 suprime; implementação nossa. Nas
  detecções públicas é quase inócuo (o MOTChallenge já as suprimiu). O
  Faster R-CNN do torchvision é o modelo pronto (permitido) e roda o NMS
  interno dele; o nosso vem por cima.
- **Limiar de score.** Detecções públicas entram como vieram — no DPM isso
  inclui os ~41% de scores negativos (score não calibrado; cortar em 0
  jogava parte do detector fora do AP). O torchvision entra com score
  ≥ 0,8 (sweep no treino: 0,3 → 0,366, 0,8 → 0,461, 0,9 → 0,433), no
  script e também no `inferencia.ipynb`.
- **Associação / nascimento / morte (Parte 1).** Compara com a última caixa
  observada (sem modelo de movimento). Hungarian. Qualquer detecção sem par
  nasce id novo. A track só é emitida depois de `min_hits` matches
  consecutivos; os quadros anteriores são descartados (online); com
  `min_hits=2` cai só o quadro de nascimento. Sem match não se escreve a
  caixa velha. A track sobrevive a `max_age` quadros sem match e morre no
  seguinte.
- **Piso fácil (Parte 0.4).** Mesmo tracker, detecções = caixas verdadeiras
  sem id. 5 elipses lentas, sem oclusão: IDF1 0,989 (o que falta é o quadro
  de nascimento de cada track), zero switches, 3 seeds. Densidade sozinha
  não quebra; velocidade quebra (0,840 em 4 px/quadro, 0,447 em 8) — o
  deslocamento passa do portão de IoU. Figura:
  `experiments/figures/synthetic_sweep.png`.
- **Trilha B (Parte 2).** A InfoNCE comparando as saídas da GRU entre si tem
  uma solução trivial — copiar o primeiro recorte e nunca atualizar — que o
  primeiro treino achou (porta de atualização 0,987). A perda preditiva
  compara a memória com os recortes futuros através de uma cabeça de
  consulta própria: porta 0,559, e a memória segue o recorte atual (cosseno
  0,79) em vez do primeiro (0,58). Recortes com `visibility < 0,25` saem do
  treino (mostram o oclusor). No rastreamento o ganho sobre o baseline é
  quase todo da cascata (controle sem cosseno: 0,575 / 0,550 contra
  0,576 / 0,552); a memória de aparência soma pouco — no sweep do treino
  +0,002 (0,5393 → 0,5417), com os valores escolhidos na borda da grade, e
  no teste as entradas aparência e "os dois" dão o mesmo IDF1. O ResNet
  ImageNet mal separa pedestres parecidos. Em que aspecto isso melhora o
  fracasso da Parte 1: menos switches (teste 176 → 155) por recasar a
  track perdida pela última caixa — não por reconhecer a pessoa pela
  aparência. Figura lado a lado: `experiments/figures/temporal_vs_baseline.png`.
- **Eixo 3 (Parte 3).** A pergunta é quem sustenta a identidade *através*
  da oclusão, então além do IDF1 medimos, em cada oclusão
  (`visibility < 0,25`), se o mesmo id previsto está dos dois lados
  (`occlusion_survival`; só entram as oclusões casadas antes e depois).
  Nas 7 cenas (n por faixa: 195 / 203 / 87 / 69), a geometria é a única
  entrada que chega ao baseline sem memória: 1–5 quadros 60,6% vs 60,0%,
  6–20 quadros 37,6% vs 37,4% (controle 32,2%); aparência (58,9 / 33,8) e
  os dois (57,6 / 33,2) ficam perto do controle. No teste (n = 18 / 35 /
  11 / 13) a geometria segura 36,2% das oclusões de 6–20 quadros contra
  25,7% do baseline. Acima de 20 quadros nenhuma entrada passa do baseline
  (21–50: 17,2% vs 16,0% da geometria; >50: empate em 8,7%) e no teste
  nenhuma sobrevive: a janela de treino (T=16) e `max_age=20` são o teto.
  **Muda com a densidade:** nas duas cenas mais densas (02, 04) e na 05 o
  baseline, que só casa pela última caixa, é o melhor (04: 48% contra 39%
  da geometria); a geometria ganha nas de densidade média (09, 10, 11) e a
  aparência só na 13 (ônibus, oclusões curtas). Resposta: a geometria
  sustenta a identidade, e só até ~20 quadros; na multidão nenhuma entrada
  recorrente supera a caixa parada. O modelo final continua sendo o de
  aparência porque a Trilha B é a memória de aparência — a geometria é a
  resposta do eixo, não troca de trilha. Figura:
  `experiments/figures/ablation_input_survival.png` (esquerda por duração,
  direita por cena, da menos à mais densa); tabelas por split em
  `experiments/results/ablation_input_survival.csv`.
- **Galeria, horizonte e correção (Parte 4).** Os switches do modelo da
  Parte 2 (antes da correção) são classificados pelas trajetórias: troca
  (54% no treino, buraco mediano 2), não recasou (40%, mediana 3), morreu
  (6%, mediana 44). "Morreu" só conta quando o buraco passa de
  `max_age + 1` e o id antigo não saiu mais depois do sumiço — um id que
  continua saindo em outra pessoa derivou, não morreu. A galeria mostra um
  de cada, com o diagnóstico ligado à janela de BPTT (T=16) quando o buraco
  passa dela. Horizonte analítico: o gradiente cai à metade em k≈5 e
  a ~20% em k≈13 (o modelo antigo era chapado em 0,0002 — não escrevia nada
  para esquecer). Empírico (nas 7 cenas): o id sobrevive a 36% das
  oclusões comparáveis (baseline 39%), com percentil 90 = 20 = `max_age`: quem limita o horizonte
  é a morte da track. A correção ataca o "não recasou": a duplicata nascida
  no quadro do miss casava antes da track confirmada; com
  `prefer_confirmed`, treino 0,542 → 0,560 e 1244 → 1066 switches.
- **Estresse do detector (Parte 5).** Sem retreinar, em cima do modelo final
  (Trilha B + `prefer_confirmed`). O `degrade` da Parte 0.2 aceita o
  `det.txt` (`degrade_detections`): mesma semente por cena, então o descarte
  é aninhado (forte ⊂ média ⊂ leve), e cada verdadeiro mantém o próprio
  score (o FP sorteia um score real), de modo que o AP só muda pelo que foi
  estragado. mAP e IDF1 no mesmo gráfico (`experiments/figures/
  stress_detector.png`). Nas 7 cenas: leve 0,451 vs 0,470, médio 0,341 vs
  0,327, forte 0,169 vs 0,171 (modelo final vs baseline). A leitura: com
  pouca sujeira a cascata ainda casa a maior parte por IoU e o estágio de
  aparência passa a aceitar recortes ruidosos e FPs como reaparecimento;
  com mais descarte o baseline perde a track e o modelo, que ainda tenta
  recasar depois do miss, segura melhor. É hipótese sobre o mecanismo, não
  medida separadamente.
- **MOTA.** Opcional e reportada à parte: com o detector congelado, os
  termos de FP/FN quase não variam entre as configurações e ela esconde o
  que muda.
- **Inferência em janelas (pergunta da Parte 2).** O que quebra na fronteira
  entre janelas: o estado da GRU recomeça do zero, toda track da janela
  seguinte nasce com id novo, e quem está ocluído atravessando a fronteira
  perde a identidade — o mesmo problema da fusão entre tiles do mosaico do
  PA1, só que no tempo. O que a Trilha B permite: sobrepor as janelas (por
  exemplo 8 quadros) e, na sobreposição, casar os ids da janela k+1 com os
  da k pelo cosseno entre a memória das tracks da janela k e a consulta das
  detecções da k+1, com o mesmo portão geométrico — a memória é um vetor
  pequeno por track, então dá para carregá-la (e a última caixa) de uma
  janela para a outra sem guardar os quadros. Sem implementação, como o
  enunciado pede; `track_sequence` processa a sequência inteira.
- **Fora do escopo.** Trilha A, filtro de Kalman (permitido só como
  comparação; `compare_kalman.py` ficou no esqueleto) e a queda de taxa de
  quadros da Parte 5.
