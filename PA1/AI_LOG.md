# AI_LOG

Uso de IA no PA1.

## Como a IA foi usada

### Ferramenta

Claude Code; os treinos rodaram no Google Colab (Tesla T4) e a IA não teve acesso ao Colab e não executou nenhum treinamento.

### Template do notebook a partir do PDF

Pedimos a leitura do enunciado e a geração de um notebook com uma célula por
questão, para organizar o trabalho. Serviu como esqueleto inicial.

### Ambiente

Checagem de que `scipy.ndimage`, `skimage.segmentation` e `sklearn.cluster` importavam, e
geração do `requirements.txt`.

### Otimização de métrica que não escalava

`instance_map` recalculava a matriz de IoU a cada limiar, e os scripts de
avaliação chamavam `instance_precision` mais dez vezes: vinte matrizes por
imagem, cada uma um laço sobre pares de máscaras booleanas de imagem inteira. No
mosaico da Parte 4 isso dava 30 s por matriz e travava a análise. Reescrito com
`np.bincount`; verificado que o resultado é idêntico ao anterior (diferença
máxima 0,00) e ~1500× mais rápido.

### Diagnóstico da Parte 5

Na pior imagem do teste o modelo previa zero núcleos contra 78 no ground
truth, mas o mapa intermediário de fronteira mostrava um anel nítido em volta de
cada núcleo, pois a rede tinha encontrado todos. A investigação mostrou a causa em
`src/dataset.py`: o alvo de 3 classes é construído com
`binary_erosion(mask, iterations=2)`, e um núcleo de 4 px de diâmetro perde o
interior inteiro. Naquela imagem, 63% dos núcleos ficam sem nenhum pixel de
interior no alvo (91 pixels de interior contra 1106 de fronteira). No dataset
inteiro: 100% dos núcleos com diâmetro ≤ 4 px e 29,8% dos de 4–8 px. Sem
interior não há marcador para o watershed, e o objeto não pode ser emitido.

A correção de pós-processamento sugerida pelo diagnóstico (semear um marcador no
ponto mais interno de cada componente sem marcador) foi implementada e
piorou o mAP médio em 0,021, ainda que resolvesse o caso alvo. Isso está
reportado como está, porque é o que o PDF pede quando a correção não funciona.
A correção que de fato resolve é no alvo (`--adaptive-boundary`).

### Elaboração das seções acima deste relatório

Pedimos que resumisse os pedidos e correções feitas e usamos o produto desse pedido como 
base deste documento.

### Erros da IA que tivemos que corrigir 

Ao longo do desenvolvimento, algumas sugestões pedidas para o modelo não foram acatadas, por estarem equivocadas, como
o uso de apenas um seed nos experimentos, que também poderia alterar a divisão dos dados e a sugestão de ajustar 
thresholds no pós-processamento da abordagem por bordas, mas essa etapa foi considerada desnecessária para os requisitos
 da atividade. 
