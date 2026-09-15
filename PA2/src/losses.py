"""
Perdas do modelo temporal (Parte 2).

Trilha A: smooth-L1 sobre a caixa prevista; opcionalmente log-verossimilhança
gaussiana quando o modelo também prevê a incerteza.

Trilha B: contrastiva ou triplet sobre as identidades do ground truth.
"""


class SmoothL1BoxLoss:
    def __init__(self, beta=1.0, parameterization="cxcywh"):
        raise NotImplementedError


class GaussianNLLBoxLoss:
    """
    -log N(caixa_verdadeira | mu, sigma). O sigma aprendido vira o portão
    adaptativo em src/association.gate.
    """

    def __init__(self, min_sigma=1e-3):
        raise NotImplementedError


class TripletIdentityLoss:
    def __init__(self, margin=0.3):
        raise NotImplementedError


class ContrastiveIdentityLoss:
    def __init__(self, temperature=0.07):
        raise NotImplementedError
