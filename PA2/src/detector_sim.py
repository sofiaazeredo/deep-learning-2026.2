"""
Simulador de detector (Parte 0.2) e teste de estresse de detecção (Parte 5).

Recebe as caixas verdadeiras e as estraga de propósito. É o mesmo código nas
duas partes: no sintético serve para validar a Parte 1 antes de baixar o
MOT17, no MOT17 serve para medir se o modelo temporal absorve ou amplifica a
falha do detector.
"""


def degrade(
    boxes,
    drop_rate=0.0,
    coord_noise=0.0,
    false_positive_rate=0.0,
    seed=None,
):
    """
    Descarta p% das caixas, adiciona ruído gaussiano nas coordenadas e injeta
    falsos positivos. Devolve caixas sem identidade — o simulador não vaza ID.
    """

    raise NotImplementedError


INTENSITIES = {
    "leve": {},
    "media": {},
    "forte": {},
}
