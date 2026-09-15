"""
Encoder de aparência (Trilha B da Parte 2, Eixo 3 da Parte 3).

Recorte da caixa -> embedding D-dimensional, com um encoder pequeno e
pré-treinado (permitido pelo enunciado). O encoder fica congelado; quem
aprende é o agregador recorrente e a perda contrastiva.
"""


def crop(frame, box, output_size=(128, 64)):
    raise NotImplementedError


class CropEncoder:
    def __init__(self, backbone="resnet18", embed_dim=128, freeze=True):
        raise NotImplementedError

    def embed(self, crops):
        """
        Lote de recortes -> embeddings L2-normalizados.
        """

        raise NotImplementedError
