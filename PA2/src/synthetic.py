"""
Gerador de vídeos sintéticos (Parte 0.1).

Vídeos 128x128 de 30 a 60 quadros, 5 a 15 elipses em movimento, tamanhos,
ruído e contraste variáveis. As elipses são desenhadas com ordem de
profundidade, então uma passa atrás da outra e realmente desaparece — sem isso
não há oclusão de verdade, só sobreposição.

Parâmetros expostos (exigidos pelo enunciado): número de objetos, velocidade
típica e duração da oclusão.
"""


def make_sequence(
    n_frames=45,
    n_objects=8,
    speed=2.0,
    occlusion_frames=10,
    size=128,
    noise=0.05,
    contrast=1.0,
    seed=None,
):
    """
    Devolve (frames, tracks): frames é (T, H, W) e tracks é a lista de caixas
    com identidade por quadro, no formato (frame, id, x, y, w, h, visibility).

    Objetos totalmente ocluídos não entram no ground truth daquele quadro.
    """

    raise NotImplementedError


def depth_order(objects):
    """
    Ordem de desenho (fundo -> frente). Quem está na frente oclui quem está
    atrás; é isso que torna a oclusão verificável.
    """

    raise NotImplementedError


def occlusion_durations(tracks):
    """
    Duração de cada buraco de visibilidade por identidade. Usado na Parte 4
    para comparar o horizonte de memória com a distribuição do dataset.
    """

    raise NotImplementedError
