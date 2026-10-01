# AI_LOG

Uso de IA no PA2.

## Como a IA foi usada

### Ferramenta

Claude Code. Os primeiros treinos rodaram no Google Colab (Tesla T4), sem
acesso da IA. Na revisão final (seção "Auditoria e revisão final") a IA
rodou os treinos, os sweeps e as avaliações localmente, numa RTX 3050, e
todo número do README saiu desses scripts.

### Estrutura do projeto a partir do PDF

Pedimos a leitura do enunciado e a geração do esqueleto de `src/` e `scripts/`,
um script por parte, espelhando a organização do PA1.

### Parte 1 — baseline

A IA implementou `dataset`, `detection` (NMS + leitura de `det.txt`),
`association`, `tracker`, `average_precision`, o drop de distractores, os
testes e os scripts a partir do enunciado e das regras BASE-01…09. Os
números (AP dos três `det.txt`, sweep no treino, IDF1) foram gerados à
mão pelos scripts, não inventados.

Decisões que a IA não tomou sozinha: split 09+11, eixo de oclusão,
comparar com a última caixa observada, Hungarian default, `min_hits`
consecutivo, sem coasting. O sweep escolheu IoU 0,2 / min_hits=2 /
max_age=20 no treino; SDP ganhou AP e IDF1.

### Parte 0.4 — piso fácil e varredura

O script reusa o `Tracker` da Parte 1 e o `degrade` sem estragar as
caixas. A varredura de oclusão com 8 objetos mal mexia no IDF1 (um alvo
escondido some no meio de sete identidades perfeitas); reduzimos para 3
objetos para o botão ficar visível. Quem quebra de verdade o baseline é
a velocidade, não o número de elipses lentas.

### Parte 2 — Trilha B

A IA implementou o encoder congelado, o cache de embeddings, o GRU, a
perda InfoNCE, o BPTT truncado e o casamento por cosseno no tracker
já existente. A fonte de detecções e as regras IoU / min_hits / max_age
ficaram congeladas.

Três bugs no primeiro rascunho: a perda contrastiva olhava um quadro
de cada vez (cada id aparece uma vez, perda zero); o InfoNCE era um
laço Python e cada época levava minutos; o cosseno comparava a
memória da GRU com o embedding cru do ResNet e o limiar 0,8 aceitava
quase qualquer par (IDF1 0,12). A associação ficou em dois estágios
(IoU nas tracks vivas, aparência só depois do miss) e a consulta
passa por um passo de GRU a partir do estado zero. (Esta versão da
Trilha B foi substituída na auditoria — ver "Auditoria e revisão final".)

### Parte 3 — Eixo 3

A IA implementou `MotionRNN`, `FusionRNN`, smooth-L1 na caixa e o
casamento depois do miss por braço (cosseno / IoU prevista / min dos
dois). Os 9 treinos (3 entradas × 3 seeds) rodaram na RTX 3050 local,
~46 min. Os números saíram dos CSVs, não foram inventados.

Decisão que a IA não tomou sozinha: o eixo (input, não célula). Na
versão final (depois da auditoria) o braço "os dois" casa pelo menor dos
dois custos — um OU entre cosseno e IoU prevista — e não herda o melhor
de cada um. A resposta do eixo saiu da sobrevivência através da oclusão:
a geometria é a única entrada que chega ao baseline, só até ~20
quadros, e nas cenas mais densas (02, 04) nenhuma entrada recorrente
supera o baseline.

### Parte 4 — galeria, horizonte e correção

A IA implementou o perfil `||∂L_t/∂h_{t-k}||`, os intervalos de
oclusão, a galeria a partir dos ID switches do CLEAR MOT e a primeira
correção (`miss_prefer_iou`, IoU da última caixa depois do miss).

A primeira galeria pegou os três gaps de 200+ quadros (pessoa que
saiu da cena); a galeria passou a escolher um switch de cada tipo. O
InfoNCE só no último quadro dava perda zero (cada id uma vez); L_t
passou a ser a InfoNCE das consultas do quadro t contra o histórico
destacado, para o único caminho até h_{t-k} ser a recorrência.

Na auditoria a `miss_prefer_iou` virou padrão do tracker da Parte 2, e a
correção da Parte 4 passou a ser `prefer_confirmed`, decidida no treino.
Números atuais (README): correção no treino 0,542 → 0,560 (1244 → 1066
switches), no teste 0,576 → 0,578; horizonte empírico 36% das oclusões
comparáveis (baseline 39%).

### Parte 5 — qualidade do detector

A IA ligou o `degrade` já testado no sintético ao `det.txt` do SDP
(`degrade_detections`), embedou as caixas novas (o cache do SDP não
serve) e comparou o modelo final com o baseline nas mesmas
detecções. Os números saíram do script, ~5 min na 3050.

Decisão que a IA não tomou sozinha: o eixo (detector, não taxa de
quadros). O temporal amplifica a falha — a resposta não foi
inventada para ficar bonita.

### Erros da IA que tivemos que corrigir

- O gráfico do descolamento saiu ordenado por número da cena (02…13),
  não por oclusão: o pandas leu `"02"` do CSV como inteiro e o
  `map` do eixo falhou. Corrigido com `zfill(2)` antes de ordenar.

## Auditoria e revisão final

Com o projeto "pronto", pedimos à IA uma inspeção completa: procurar erro de
lógica, violação das regras do enunciado e conferir se os números fazem
sentido.

### Como a auditoria foi feita

Quatro revisores em paralelo (métricas e protocolo; detecção, associação e
tracker; modelo temporal; entregáveis) leram o código sem editar nada. Para
checar a implementação, a IA rodou as ferramentas **proibidas** num ambiente
separado, fora do repositório e fora do `.venv` do projeto, só como gabarito
sobre as MESMAS entradas:

- `torchvision.ops.nms` vs o nosso NMS: 0 diferenças em 47.793 quadros;
- `pycocotools` / VOC de referência vs o nosso AP: idênticos a 4 casas;
- `motmetrics` vs o nosso IDF1 / switches / fragmentações / MOTA: idênticos
  nas 14 execuções salvas;
- TrackEval (o avaliador oficial do MOTChallenge): diferenças pequenas que
  apontaram dois desvios de protocolo (abaixo);
- SORT e ByteTrack nas nossas detecções: o baseline ingênuo empatou com o
  SORT original (IDF1 0,536), o que deu confiança na Parte 1 — e o modelo
  temporal ficou ABAIXO de tudo, o que denunciou o problema da Trilha B.

Nenhuma dessas bibliotecas entrou no projeto.

### O que estava errado e foi corrigido

- **Distractores.** Tratávamos toda linha não-pedestre como distractor
  (carro, oclusor…) e casávamos a predição só contra eles. O protocolo do
  TrackEval usa só as classes 2, 7, 8 e 12 e casa contra todo o gt do
  quadro. Depois da correção: IDF1, switches, fragmentações, MOTA, FP e FN
  idênticos ao TrackEval nas 7 cenas.
- **Continuidade do CLEAR MOT** só do quadro anterior (antes: do último par
  de sempre, o que deixava um gt que voltava "roubar" a predição).
- **A GRU copiava o primeiro recorte.** Porta de atualização 0,987, memória
  0,997 parecida com o primeiro recorte, perda no piso teórico — e desligar o
  cosseno *melhorava* o IDF1. A InfoNCE comparava a memória com as próprias
  saídas da GRU, e copiar o primeiro recorte a minimizava. Nova perda
  (`PredictiveContrastiveLoss`): a memória depois de t tem que reconhecer os
  recortes FUTUROS da mesma pessoa, com uma cabeça de consulta própria.
  Porta 0,559, memória segue o recorte atual (0,79) e não o primeiro (0,58).
- **Portão de IoU** que o enunciado pede sobre o cosseno não existia depois
  do miss. Entrou, com a cascata (IoU nas tracks perdidas antes do cosseno)
  como padrão; limiar e portão escolhidos por sweep **só no treino**.
- **A geometria do Eixo 3 perdia para "copiar a última caixa"** (2,3 a 11×
  pior): regredia a caixa normalizada, onde um quadro de movimento é ~0,001.
  Passou a prever o deslocamento codificado (como o Faster R-CNN) com a
  velocidade na entrada: 0,62–0,78× o erro de copiar.
- **Entregáveis:** `src/inference.py` estava no esqueleto (o notebook não
  rodava) e o checkpoint não estava no repositório. Os dois entraram.
- **Segunda fonte de detecção** (torchvision) nunca tinha rodado; rodou, com
  o limiar de score escolhido no treino.
- **Parte 5:** o `degrade` jogava fora o score real do detector (AP caía
  mesmo sem estragar nada) e a porcentagem do README misturava splits.
- **Parte 4:** a antiga correção virou parte do tracker da Parte 2, então a
  IA mediu os tipos de ID switch do modelo da Parte 2: "não recasou" é o
  segundo tipo (40% no treino, buraco mediano de 3 quadros, com a
  classificação corrigida na revisão final) — a duplicata que nasce no
  quadro do miss ganha da track verdadeira. A correção (`prefer_confirmed`)
  saiu desse diagnóstico e foi decidida no treino.

### Segunda revisão (antes da entrega)

Dois revisores em paralelo (código; números, README e cobertura do
enunciado) e a IA de novo com TrackEval e motmetrics fora do projeto, sobre
todas as 17 execuções salvas. O que mudou:

- `failure_correction.py` gravava as trajetórias já sem distractores
  (quem reavaliasse o CSV os tirava duas vezes); agora grava as cruas, como
  os outros scripts, e o CSV reavaliado bate com o relatado.
- O `inferencia.ipynb` sem `det.txt` rodava o Faster R-CNN com score 0,5,
  não o 0,8 escolhido no treino; corrigido, com teste.
- A classificação dos switches contava como "morreu" um buraco de
  `max_age + 1` (a track ainda casa nesse quadro) e ids que continuavam
  saindo em outra pessoa; corrigida, com teste — as fatias passaram de
  54/34/12% para 54/40/6%. Os diagnósticos citam a janela de BPTT quando o
  buraco passa de T=16.
- A sobrevivência do Eixo 3 saiu por split (treino / teste / 7 cenas), com
  n por faixa e um painel por cena ordenado por densidade.
- README: a resposta da Parte 5 é "amplifica" (o "depende da intensidade"
  só valia nas 2 cenas de teste e invertia no treino); a decisão do
  detector passou a citar só o treino; números do DPM sem arquivo que os
  reproduzisse saíram; a pergunta das janelas da Parte 2 ganhou resposta.
- Única diferença restante para o TrackEval: num quadro sem nenhuma caixa
  prevista ele não zera a memória do quadro anterior e conta menos
  fragmentações (documentado no README; a nossa contagem bate com o
  motmetrics).

### Erros da IA nesta revisão

- Matou um treino que estava na época 13 de 15: o filtro do log mostrava só
  as épocas 1 e 15 e parecia travado. O usuário apontou que a estimativa de
  tempo estava errada — estava: cada treino levava ~23 min, não 4, porque a
  perda da caixa era codificada par a par. Vetorizada (172 → 30 ms/passo).
- `pgrep -f` / `pkill -f` casaram o próprio comando que esperava pelo
  processo (duas vezes).
- A GPU caiu no meio da Parte 5 ("GPU requires reset"); a Parte 5 rodou de
  novo.
- Estimou tempo antes de medir; depois disso passou a medir a primeira época
  antes de prometer um horário.

