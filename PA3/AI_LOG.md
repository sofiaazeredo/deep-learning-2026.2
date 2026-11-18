# AI_LOG — como usamos IA neste PA

> RASCUNHO: revisem e reescrevam com as palavras de vocês. Só deixem episódios que realmente aconteceram.
> A política do PA permite IA; o que não vale é entregar algo que vocês não entendem.

## Ferramenta
- Claude (Anthropic), pela interface web/app. Nenhuma API de LLM foi usada para rotular ou classificar imagens.

## O que a IA gerou (esqueleto do repositório)
- Código-base de `src/` (Parte 0, FixMatch/Trilha B, sweep, gráficos, Grad-CAM), `predict.py`, `tests/` e o notebook do Colab.
- Nós **não** rodamos esse código nos dados reais antes de receber: a IA o testou só em imagens sintéticas, em CPU.
  Todos os números do PA vêm das nossas execuções no Colab.

## Episódios (preencham)
1. **Hiperparâmetros chutados.** O primeiro rascunho tinha lr/épocas/τ fixos no código. Pedimos para tirar tudo do código;
   passou a existir `configs/` + `src/sweep.py`, que escolhe só no val. *(Contem o que mudou nos resultados.)*
2. **Bug de dtype no pandas.** A detecção de colunas de `classes.csv` falhava com pandas novo (texto não é mais `object`).
   Foi pego pelo `tests/check_data.py` rodando em dados sintéticos.
3. **Teste de overfit mal desenhado.** O primeiro `check_pipeline` falhou; investigamos e vimos que o loop estava certo
   e a aumentação é que impedia chegar a 100% de treino. O teste passou a rodar sem aumentação.
4. **Premissa errada sobre os dados.** A IA assumiu que existia um `train.csv`; o README do pacote mostrou que o pool é a pasta
   `images/train/`. O `check_data` acusou o FAIL no primeiro uso real; o código passou a ler o pool da pasta, e o gerador de dados
   sintéticos ganhou o layout real (`--real_layout`) para testar esse caminho.
5. *(adicionem: onde a IA errou/acertou nos resultados de vocês, o que vocês verificaram à mão, o que descartaram)*

## O que vocês precisam conseguir explicar sem a IA
- Por que a época/τ/correção foram escolhidos no val e não no test_public.
- O que o alinhamento de distribuição e o limiar por classe fazem com os pseudo-rótulos.
- Por que a vista forte do FixMatch parte do mesmo recorte da fraca (`src/fixmatch.py`, `UnlDS`).
- Como o Grad-CAM é calculado (`src/saliencia.py`).
