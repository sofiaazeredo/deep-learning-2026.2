"""
Modelo temporal (Parte 2) e as células comparadas no Eixo 1 da Parte 3.

Trilha A — RNN como modelo de movimento: o estado recorrente recebe a última
observação (caixa, opcionalmente confiança e dt) e prevê a caixa do quadro
seguinte. Sob oclusão roda para frente sem observação.

Trilha B — RNN como memória de aparência: um agregador recorrente mantém o
estado de aparência da track, atualizado a cada observação.

A escolha da trilha é única e fica registrada no README.
"""

CELLS = ("rnn", "lstm", "gru")


class MotionRNN:
    """
    Trilha A. `cell` em CELLS, com orçamento de parâmetros aproximadamente
    igual entre as células (Eixo 1). `predict_sigma=True` emite também a
    incerteza, que vira portão de associação adaptativo.

    `bidirectional=True` é o modo offline do Eixo 4 — não pode ser usado na
    avaliação online.
    """

    def __init__(self, cell="gru", hidden=128, predict_sigma=False,
                 bidirectional=False):
        raise NotImplementedError


class AppearanceRNN:
    """
    Trilha B. Agrega embeddings de recortes ao longo do tempo num estado de
    aparência por track.
    """

    def __init__(self, cell="gru", embed_dim=128, hidden=128):
        raise NotImplementedError


def build_model(name, **kwargs):
    """
    Registro nome -> modelo, e loader de checkpoint (mesmo padrão do PA1).
    """

    raise NotImplementedError


def load_checkpoint(path, device="cpu"):
    raise NotImplementedError
