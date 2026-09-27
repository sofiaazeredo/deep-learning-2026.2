# AI_LOG

Uso de IA no PA2.

## Como a IA foi usada

### Ferramenta

Claude Code; os treinos rodaram no Google Colab (Tesla T4) e a IA não teve
acesso ao Colab e não executou nenhum treinamento.

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
passa por um passo de GRU a partir do estado zero.

### Erros da IA que tivemos que corrigir

- O gráfico do descolamento saiu ordenado por número da cena (02…13),
  não por oclusão: o pandas leu `"02"` do CSV como inteiro e o
  `map` do eixo falhou. Corrigido com `zfill(2)` antes de ordenar.
