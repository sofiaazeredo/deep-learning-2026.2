# AI_LOG

Como usamos IA no PA1.

Ferramentas: ChatGPT/Claude durante a construção do projeto e **Claude Code**
(Anthropic, rodando em terminal com acesso ao repositório) na segunda metade.
Todos os treinos rodaram no Google Colab (Tesla T4) e na máquina local.

Organizamos por etapa, dizendo em cada uma **o que nós decidimos** e **o que a
IA fez**.

---

## Parte 0 — Teste unitário sintético

**Nós decidimos:** que o teste tinha que exercitar o mesmo pipeline dos dados
reais (mesmo encoder-decoder, mesma perda, mesma decodificação, mesma métrica),
senão não testaria nada.

**A IA fez:** o gerador (`src/synthetic.py`) e o script de treino
(`scripts/train_synthetic.py`), a partir dessa especificação. A escolha de
manter as elipses sem sobreposição — o pixel disputado fica com a instância
colocada primeiro, reproduzindo a propriedade do DSB2018.

**O que o teste encontrou:** a perda virava `NaN` por volta da época 10 e o mAP
ia a zero. A IA investigou e mostrou que o forward estava correto (perda 0,244,
logits no máximo 6,17) mas o gradiente vinha `inf`, e que o mesmo código com a
mesma seed rodava limpo na CPU. Ou seja: **a GPU local estava defeituosa**, não o
código. Foi o teste unitário justificando a própria existência logo na primeira
execução. A IA também fixou `cudnn.deterministic` e a seed do DataLoader, porque
sem isso o erro era intermitente (0,438 numa execução, 0,000 na seguinte) e
impossível de diagnosticar, e acrescentou uma guarda que aborta se a perda virar
NaN em vez de reportar uma métrica sem sentido.

---

## Parte 1 — Baseline de segmentação semântica

**Nós decidimos:** usar U-Net (arquitetura da aula, e o paper original é
justamente de segmentação de células), saída de 2 canais com softmax, extração
ingênua por limiar + componentes conexos, e casamento **guloso por IoU
decrescente** com AP = TP/(TP+FP+FN), que é a definição do DSB2018.

**A IA fez:** _(preencher — o que ela gerou na primeira metade: `DSB2018Dataset`,
loop de treino, `iou_score`/`dice_score`, o matching de instâncias?)_

Na segunda metade, sobre este mesmo código, a IA reescreveu
`instance_iou_matrix` usando `np.bincount` sobre os pares de rótulos, no lugar do
laço que comparava cada máscara verdadeira com cada máscara prevista, e
acrescentou `instance_scores`, que calcula a matriz de IoU **uma vez por imagem**
e reaproveita nos dez limiares. A versão anterior recalculava a matriz a cada
limiar, e os scripts de avaliação ainda chamavam `instance_precision` mais dez
vezes: vinte matrizes por imagem. Na escala do mosaico da Parte 4 isso dava 30 s
por matriz e travava a análise. Antes de trocar, a IA verificou que o resultado
novo é **idêntico** ao antigo (diferença máxima 0,00) em casos aleatórios e
degenerados. Também ajustou `scripts/evaluate.py` e `scripts/evaluate_boundary.py`
para usar esse caminho único.

---

## Parte 2 — Cabeça de instâncias (Trilha A)

**Nós decidimos:** a Trilha A (fronteira + watershed), o alvo de 3 classes
(fundo / interior / fronteira), a construção da fronteira por erosão
morfológica, a largura de 2 px, o peso 2,0 na classe fronteira e a decodificação
por watershed com os interiores como marcadores sobre a transformada de
distância.

**A IA fez:** _(preencher)_

---

## Parte 3 — Ablações

**Nós decidimos:** os dois eixos (Eixo 1, recuperação de resolução; Eixo 2,
função de perda), quais arquiteturas comparar (`UNet`, `UNetNoSkips`,
`UNetASPP`), as perdas e os valores de γ ∈ {1, 2, 5}, as 2 seeds (42 e 123), e
fixar `SPLIT_SEED = 42` independente da seed de treino para que todas as
execuções compartilhem exatamente o mesmo conjunto de teste.

**A IA fez:** `scripts/summarize_resolution_ablation.py`, a partir do
`summarize_loss_ablation.py` que já tínhamos.

---

## Parte 4 — Inferência em mosaico

**Nós decidimos** a abordagem inteira: que o objeto na costura vira dois porque
os IDs de instância são locais a cada tile, que a correção seria **fusão de
instâncias entre tiles por IoU na faixa de sobreposição**, e escrevemos o núcleo
dela em `src/mosaic.py` — `generate_tiles`, `mask_iou` e `merge_tile_instances`.
É a decisão de projeto da Parte 4, e é nossa.

**A IA fez:** o driver que faltava (`scripts/mosaic_inference.py`) — montar o
mosaico deslocando os rótulos de cada imagem para IDs globais, rodar as duas
decodificações e medir. Ela sugeriu acrescentar uma terceira decodificação, a
passada única sobre o mosaico inteiro, como referência para saber quanto da
perda vem do tiling e quanto é do modelo; e selecionar para a figura o núcleo
que a versão sem fusão mais fragmenta. Resultado: 0,2641 → 0,3350 de mAP com a
nossa fusão.

---

## Parte 5 — Galeria de falhas

**Nós decidimos:** usar as 5 piores imagens por mAP e mostrar o mapa de fronteira
como mapa intermediário.

**A IA fez, e aqui ela foi além de gerar código:** o diagnóstico. Na pior imagem
o modelo previa **0 núcleos** contra 78 do ground truth, mas o mapa de fronteira
mostrava um anel nítido em volta de cada um — a rede tinha achado todos. A IA
rastreou até `_create_boundary_target`: a erosão de 2 px que constrói a classe
fronteira apaga o interior inteiro de núcleos pequenos. Mediu: naquela imagem
63% dos núcleos ficam sem nenhum pixel de interior (91 pixels de interior contra
1106 de fronteira); no dataset inteiro, 100% dos núcleos com diâmetro ≤ 4 px e
29,8% dos de 4–8 px. Sem interior não há marcador para o watershed e o objeto
não pode sair.

A IA também calculou o campo receptivo camada a camada (200 px sem ASPP, 456 px
com) e comparou com os tamanhos dos núcleos (12 px de mediana, 83 px no máximo),
concluindo que **0,00% dos núcleos excedem o campo receptivo** — ou seja, o
exemplo do enunciado não se aplica ao nosso caso.

A correção de pós-processamento sugerida pelo diagnóstico foi implementada e
**piorou** o mAP médio em 0,016, embora resolvesse o caso alvo. Está reportada
assim, como o PDF pede. A correção que de fato resolve é no alvo
(`--adaptive-boundary`, que leva os núcleos sem interior de 316/4692 para 0).

---

## Parte 6 — Teste de estresse

**Nós decidimos:** que faríamos o teste de estresse, e tínhamos pensado em variar
densidade/espaçamento em cenas sintéticas.

**A IA apontou** que isso não é nenhuma das três opções do enunciado (modalidade,
corrupções, escala) e implementou as duas que rodam só com inferência:
corrupções em 3 intensidades e mudança de escala. Aceitamos a correção.

---

## Parte 7 — Pipeline de inferência

**Nós decidimos:** o entregável (imagem qualquer → máscara colorida + contagem,
sem retreinar).

**A IA fez:** `src/inference.py` e `inferencia.ipynb`, incluindo detectar sozinho
se o checkpoint é o baseline de 2 canais ou o modelo de fronteira de 3, e
devolver a máscara na resolução original da imagem.

---

## O que a IA NÃO fez

- Não rodou nenhum treinamento: todos foram executados por nós, no Colab e na
  máquina local.
- Não escolheu a trilha da Parte 2 nem os eixos da Parte 3.

## Erros da IA que tivemos que corrigir

- Numa verificação, executou `scripts/train.py` sem querer (o script não tem
  `--help`, então rodar para "testar o import" dispara um treino) e sobrescreveu
  `experiments/results/training_history.csv`. Restaurado do git.
- Sugeriu uma célula de setup que clonava o repositório por cima de si mesmo,
  criando `PA1/deep-learning-2026.2/` na branch errada e quebrando o import.
- Deixou um typo (`script/` em vez de `scripts/`) numa célula que passou para nós.
- Gerou o primeiro notebook com as linhas de código sem quebra de linha no fim,
  de modo que cada célula viraria uma linha só — a primeira célula inteira teria
  virado comentário.
- Escreveu `scripts/mosaic_inference.py` sem perceber que a métrica de instância
  daquela época era quadrática; o script travou na primeira execução e só depois
  ela foi otimizar a métrica.
- Na primeira versão, a figura da costura escolhia o maior núcleo que cruzava a
  fronteira, e não o que tinha sido de fato partido — a figura não mostrava a
  falha que deveria ilustrar.
- O `stress_test.py` imprimia as quedas de mAP com sinal trocado (`+10,1%` para
  uma queda de 10,1%).
- Ao inserir os resultados das Partes 4 a 6 no README, colou o bloco na seção
  errada, depois dos comandos de reprodução em vez de em "Resultados".
- Afirmou que o template do notebook estava todo quebrado quando só as três
  células que ela tinha acabado de inserir estavam.
- Errou estimativas de tempo em ordens de grandeza nos dois sentidos: previu
  60–75 min para a execução local das Partes 4 a 6, que levou cerca de 20.

## Limitações conhecidas do que entregamos

- O split original usava `random_split`, sem estratificação por modalidade, que é
  o que o enunciado pede. Corrigimos: `create_splits` agora estratifica por uma
  modalidade derivada de duas estatísticas da imagem. Ao medir, descobrimos que o
  sorteio aleatório já tinha caído praticamente balanceado (81,7/15,7/2,6% no
  treino antigo contra 81,5/16,0/2,4% no estratificado), então a correção vale
  pela conformidade e pela justificativa, não porque a avaliação estivesse
  enviesada.
- A IA chegou a afirmar, e escrevemos no README, que o modelo ia mal nas
  modalidades claras "porque são minoria no treino". A medição desmentiu: elas
  são ~18% do treino nos dois splits. A correlação de −0,61 entre mAP e brilho
  vem da dificuldade dessas modalidades, não da ausência delas no treino.
- Todas as imagens são redimensionadas para 256×256, inclusive as de 1024×1024 e
  1040×1388, e as métricas são medidas nessa resolução.
