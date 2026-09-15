"""
Inferência numa sequência qualquer (entregável inferencia.ipynb).

Recebe o caminho de uma sequência, devolve o vídeo com as identidades
coloridas de forma consistente e a contagem de objetos únicos. Roda sem
retreinar.
"""


def track_sequence(sequence_path, checkpoint=None, detector="SDP",
                   device="cuda"):
    """
    Devolve um dicionário com:
      tracks    trajetórias previstas (frame, id, x, y, w, h)
      count     número de objetos únicos no vídeo
      frames    quadros anotados, coloridos por identidade
    """

    raise NotImplementedError


def draw_tracks(frame, boxes, ids):
    """
    Uma cor estável por identidade, para o olho conferir o que a métrica diz.
    """

    raise NotImplementedError


def write_video(frames, path, fps=30):
    raise NotImplementedError


def windowed_inference(sequence_path, window=64, overlap=8, **kwargs):
    """
    Um vídeo de duas horas não cabe na memória, então a inferência roda em
    janelas de T quadros. A costura de identidades entre janelas é o análogo
    temporal da fusão entre tiles do mosaico do PA1 (pergunta da Parte 2).
    """

    raise NotImplementedError
