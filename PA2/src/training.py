"""
Regimes de treino do modelo temporal (Eixo 2 da Parte 3).

Na inferência o modelo se alimenta das próprias previsões — e, sob oclusão, só
delas. Se nunca viu isso no treino, a distribuição muda debaixo dele. Os três
regimes existem para medir exatamente esse descolamento.
"""

REGIMES = ("teacher_forcing", "scheduled_sampling", "free_running")


def sampling_probability(regime, epoch, total_epochs):
    """
    Probabilidade de alimentar a observação verdadeira no passo seguinte.
    Constante 1 no teacher forcing, decaindo no scheduled sampling, 0 no
    free-running.
    """

    raise NotImplementedError


def truncated_bptt(model, window, loss_fn, regime, clip_grad=None):
    """
    Um passo de BPTT truncado sobre uma janela de T quadros. `clip_grad=None`
    desliga o gradient clipping, que é uma das condições do Eixo 2.
    """

    raise NotImplementedError


def gradient_norm_profile(model, window):
    """
    Norma de dL_t/dh_{t-k} em função de k — a curva de gradiente que some, na
    forma analítica que a Parte 4 exige.
    """

    raise NotImplementedError
