"""
Gestão de tracks: nascimento, morte e o laço de rastreamento (Parte 1.2/1.4).

O mesmo laço serve ao baseline ingênuo e ao modelo temporal da Parte 2 — o que
muda é quem prevê a caixa (ou o embedding) do próximo quadro.
"""


class Track:
    """
    Estado de um objeto: identidade, última caixa, idade, quadros sem
    observação e o estado recorrente (Parte 2).
    """

    def __init__(self, track_id, box, frame):
        raise NotImplementedError


class Tracker:
    """
    Parâmetros de regra:
      iou_threshold   limiar de associação
      max_age         k quadros sem observação antes de matar a track
      min_hits        observações antes de a track ser emitida
      matcher         'greedy' ou 'hungarian'
      motion          None (baseline), 'kalman' (baseline permitido) ou o
                      modelo recorrente da Parte 2
    """

    def __init__(self, iou_threshold=0.3, max_age=30, min_hits=3,
                 matcher="hungarian", motion=None):
        raise NotImplementedError

    def update(self, detections, frame):
        """
        Um quadro: prever, associar, atualizar, nascer, morrer.
        """

        raise NotImplementedError

    def run(self, sequence):
        """
        Sequência inteira -> pred_tracks no formato que src/metrics.py consome.
        """

        raise NotImplementedError


class KalmanMotion:
    """
    Filtro de Kalman de velocidade constante. Permitido APENAS como baseline de
    comparação — não pode ser o modelo temporal da Parte 2.
    """

    def __init__(self):
        raise NotImplementedError
