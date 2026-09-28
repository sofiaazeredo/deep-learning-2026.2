"""
Simulador de detector (Parte 0.2) e teste de estresse de detecção (Parte 5).

Recebe as caixas verdadeiras e as estraga de propósito. É o mesmo código nas
duas partes: no sintético serve para validar a Parte 1 antes de baixar o
MOT17, no MOT17 serve para medir se o modelo temporal absorve ou amplifica a
falha do detector.

Entrada: linhas de ground truth de uma sequência inteira. Só são lidas a
coluna 0 (frame) e as colunas 2 a 5 (x, y, w, h), então servem tanto as
linhas do gerador sintético (frame, id, x, y, w, h, visibility) quanto as do
gt.txt do MOT17 (frame, id, x, y, w, h, conf, class, visibility).

Saída: (frame, x, y, w, h, score), sem identidade, ordenada por quadro e,
dentro do quadro, por score decrescente — a ordem de saída de um detector
depois do NMS, que não carrega nada da identidade.

Os três botões:
  drop_rate            probabilidade de cada caixa verdadeira sumir;
  coord_noise          desvio do ruído gaussiano como fração do tamanho da
                       caixa (centro em x e w pela largura, em y e h pela
                       altura), então 0.05 significa o mesmo numa elipse de
                       10 px e num pedestre de 150 px;
  false_positive_rate  falsos positivos por caixa verdadeira: cada quadro
                       recebe Poisson(taxa x caixas verdadeiras do quadro),
                       com tamanho sorteado entre as caixas verdadeiras da
                       sequência e posição uniforme na imagem.

O score de verdadeiros e falsos positivos vem da mesma distribuição, então
um limiar de score não separa os dois — quem separa é a associação. Com
`scores` (detecções reais, Parte 5) cada verdadeiro mantém o próprio score e
o FP sorteia um dos scores reais: o ranking do detector — e o AP — só muda
pelo que foi estragado. Sem `scores` (gt sintético, que não tem score) os
dois saem de uniforme(0,5; 1).
"""

import numpy as np

SCORE_RANGE = (0.5, 1.0)
MIN_SIZE = 1.0


def degrade(
    boxes,
    drop_rate=0.0,
    coord_noise=0.0,
    false_positive_rate=0.0,
    image_size=None,
    seed=None,
    scores=None,
):
    """
    Descarta p% das caixas, adiciona ruído gaussiano nas coordenadas e injeta
    falsos positivos. Devolve caixas sem identidade — o simulador não vaza ID.

    image_size = (largura, altura), obrigatório quando há falsos positivos.
    scores = score de cada caixa de entrada (detecções reais), ou None.
    """

    if not 0.0 <= drop_rate <= 1.0:
        raise ValueError(f"drop_rate precisa estar em [0, 1]: {drop_rate}")
    if coord_noise < 0:
        raise ValueError(f"coord_noise precisa ser >= 0: {coord_noise}")
    if false_positive_rate < 0:
        raise ValueError("false_positive_rate precisa ser >= 0: "
                         f"{false_positive_rate}")
    if false_positive_rate > 0 and image_size is None:
        raise ValueError("falsos positivos precisam de image_size")
    if scores is not None and len(scores) != len(boxes):
        raise ValueError("scores precisa ter um valor por caixa")

    rng = np.random.default_rng(seed)

    if len(boxes) == 0:
        return []

    frames = np.array([row[0] for row in boxes], dtype=np.int64)
    xywh = np.array([row[2:6] for row in boxes], dtype=np.float64)

    # Descarte.
    keep = rng.random(len(boxes)) >= drop_rate
    kept_frames, kept = frames[keep], xywh[keep]

    # Ruído relativo ao tamanho, aplicado no centro e no tamanho. Os sorteios
    # acontecem sempre (a sequência aleatória não depende do ruído), mas sem
    # ruído a caixa passa intacta, bit a bit.
    w, h = kept[:, 2], kept[:, 3]
    noise = rng.normal(0.0, 1.0, (4, len(kept)))

    if coord_noise > 0:
        cx = kept[:, 0] + w / 2 + noise[0] * coord_noise * w
        cy = kept[:, 1] + h / 2 + noise[1] * coord_noise * h
        new_w = np.maximum(w + noise[2] * coord_noise * w, MIN_SIZE)
        new_h = np.maximum(h + noise[3] * coord_noise * h, MIN_SIZE)
        detections = np.column_stack([cx - new_w / 2, cy - new_h / 2,
                                      new_w, new_h])
    else:
        detections = kept.copy()

    out_frames = kept_frames
    n_true = len(kept)

    # Falsos positivos, proporcionais às caixas verdadeiras do quadro (antes
    # do descarte: o detector erra onde há gente, não onde sobrou gente).
    if false_positive_rate > 0:
        width, height = image_size
        unique_frames, counts = np.unique(frames, return_counts=True)
        n_fp = rng.poisson(false_positive_rate * counts)

        fp_frames = np.repeat(unique_frames, n_fp)
        sizes = xywh[rng.integers(len(xywh), size=len(fp_frames)), 2:4]
        sizes = np.minimum(sizes, [width, height])

        fp_x = rng.uniform(0.0, 1.0, len(fp_frames)) * (width - sizes[:, 0])
        fp_y = rng.uniform(0.0, 1.0, len(fp_frames)) * (height - sizes[:, 1])

        detections = np.vstack([detections,
                                np.column_stack([fp_x, fp_y, sizes])])
        out_frames = np.concatenate([out_frames, fp_frames])

    if scores is None:
        out_scores = rng.uniform(*SCORE_RANGE, len(out_frames))
    else:
        real = np.asarray(scores, dtype=np.float64)
        out_scores = np.concatenate([
            real[keep],
            rng.choice(real, size=len(out_frames) - n_true),
        ])

    # Quadro crescente, score decrescente dentro do quadro.
    order = np.lexsort((-out_scores, out_frames))

    return [(int(out_frames[i]), *map(float, detections[i]),
             float(out_scores[i]))
            for i in order]


def degrade_detections(dets, drop_rate=0.0, coord_noise=0.0,
                       false_positive_rate=0.0, image_size=None, seed=None):
    """
    Igual ao `degrade`, mas a entrada já é (frame, x, y, w, h, score) — o
    formato do det.txt depois do NMS. A Parte 5 estraga o SDP, não o gt.
    """

    boxes = [(int(row[0]), index + 1, float(row[1]), float(row[2]),
              float(row[3]), float(row[4]))
             for index, row in enumerate(dets)]
    return degrade(boxes, drop_rate=drop_rate, coord_noise=coord_noise,
                   false_positive_rate=false_positive_rate,
                   image_size=image_size, seed=seed,
                   scores=[float(row[5]) for row in dets])


# As três intensidades do teste de estresse da Parte 5 (e da varredura do
# sintético). Valores iniciais — calibrar contra o que o detector público
# do MOT17 erra de fato.
INTENSITIES = {
    "leve": {"drop_rate": 0.1, "coord_noise": 0.02,
             "false_positive_rate": 0.05},
    "media": {"drop_rate": 0.25, "coord_noise": 0.05,
              "false_positive_rate": 0.15},
    "forte": {"drop_rate": 0.5, "coord_noise": 0.1,
              "false_positive_rate": 0.3},
}
