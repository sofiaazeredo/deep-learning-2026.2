"""
MOT17 / MOTChallenge: leitura, split e janelas temporais.

Formato do gt.txt e do det.txt:
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility

O campo visibility alimenta a análise de oclusão da Parte 4.

Split POR SEQUÊNCIA, nunca por quadro: separar quadros aleatoriamente coloca o
quadro t no treino e o t+1 na validação, e o modelo temporal seria avaliado em
cima do que praticamente já viu. Pelo menos uma sequência inteira fica fora.
"""

SEQUENCE_ROOT = "data/MOT17"
DETECTORS = ("DPM", "FRCNN", "SDP")


def read_mot_file(path):
    """
    gt.txt ou det.txt -> lista de linhas já tipadas.
    """

    raise NotImplementedError


def load_sequence(name, detector="SDP", root=SEQUENCE_ROOT):
    """
    Devolve (info, gt_tracks, detections) de uma sequência.
    """

    raise NotImplementedError


def create_splits(sequences, seed=42):
    """
    Split por sequência. O critério (câmera parada vs. móvel, densidade, ponto
    de vista) é justificado na apresentação e impresso por split_report.
    """

    raise NotImplementedError


def split_report(splits):
    """
    Imprime quantos quadros, identidades e a densidade média por split.
    """

    raise NotImplementedError


class TrackWindowDataset:
    """
    Trajetórias do ground truth fatiadas em janelas de T quadros, que é a
    unidade de BPTT truncado do treino da Parte 2 e do Eixo 1 da Parte 3.
    """

    def __init__(self, sequences, window=16, stride=1):
        raise NotImplementedError
