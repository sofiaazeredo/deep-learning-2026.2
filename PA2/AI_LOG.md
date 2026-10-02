# AI_LOG

Uso de IA no PA2.

## Como a IA foi usada

### Ferramenta

Claude Code. Os treinos rodaram no Google Colab (Tesla T4), sem
acesso da IA. l

Com o projeto "pronto", pedimos à IA uma inspeção: procurar erro de
lógica, violação das regras do enunciado e conferir se algum dos números
atenção negativamente. 

### Estrutura do projeto a partir do PDF

Pedimos a leitura do enunciado e a geração do file tree,
espelhando a organização do PA1, para inspiração.


### Parte 2 — Trilha B

Pedimos que a IA identificasse os bugs no nosso primeiro rascunho: a perda contrastiva olhava um quadro
de cada vez (cada id aparece uma vez, perda zero); o InfoNCE era um
laço Python e cada época levava minutos; o cosseno comparava a
memória da GRU com o embedding cru do ResNet e o limiar 0,8 aceitava
quase qualquer par (IDF1 0,12).

### Parte 3 — Eixo 3

A IA identificou que, na nossa implementação, o braço "os dois" (min do cosseno e da IoU prevista) casava não pelo menor dos
dois custos, mas herdava o melhor de cada um. 

### Parte 4 — galeria, horizonte e correção

A IA apontou que InfoNCE só no último quadro dava perda zero (cada id uma vez) e sugeriu que L_t
passasse a ser a InfoNCE das consultas do quadro t contra o histórico
destacado, para o único caminho até h_{t-k} ser a recorrência.

### Parte 5 — qualidade do detector

A IA comparou o modelo final com o baseline nas mesmas
detecções. Os números saíram do script, ~5 min na 3050.

### README e AI LOG

Pedimos que a IA gerasse um README para o repositório depois do fim de todas as alterações, que foi usado como base para o README final. 
Pedimos que o modelo também resumisse as suas alterações sugeridas, para que pudessemos confeccionar este documento. 

### Erros da IA que tivemos que corrigir

- A IA sugeriu uma alteração que seria equivalente ao tratamente de toda linha não-pedestre como distractor
  (carro, oclusor…) e ao casamento da predição só contra eles. A adotamos, percebemos o que ela implicava e desfizemos a
  alteração.
- Portão de IoU que o enunciado pede sobre o cosseno depois
  do miss deixou de existir depois de uma tentativa de correção. 
- Na estrutura inicial, `src/inference.py` estava no esqueleto e o checkpoint não estava no repositório. 
- A porcentagem referente ao score real do detector, no README gerado, misturava splits.
