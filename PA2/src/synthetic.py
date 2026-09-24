"""
Gerador de vídeos sintéticos (Parte 0.1).

Vídeos 128x128 de 30 a 60 quadros, 5 a 15 elipses em movimento, tamanhos,
ruído e contraste variáveis. As elipses são desenhadas com ordem de
profundidade, então uma passa atrás da outra e realmente desaparece — sem isso
não há oclusão de verdade, só sobreposição.

Parâmetros expostos (exigidos pelo enunciado): número de objetos, velocidade
típica e duração da oclusão.

Como a duração da oclusão vira um número controlado: quando
occlusion_frames = N > 0, dois dos objetos formam um par "oclusor + alvo". O
oclusor fica na frente de todos; durante N quadros centrados no vídeo o alvo
anda colado no centro dele, e o oclusor é grande o bastante para cobri-lo
inteiro (semieixo na direção do movimento = semieixo maior do alvo + folga).
Antes e depois o alvo tem velocidade própria, então entra por trás, some e
sai. O buraco medido no ground truth pode passar de N por um quadro de cada
lado (o quadro em que o alvo ainda não saiu nenhum pixel de trás do oclusor);
com speed baixa esse excesso cresce.

Os demais objetos ficam atrás do par e se ocluem entre si ao acaso — essas
oclusões naturais também aparecem em occlusion_durations.

Convenções: quadros indexados a partir de 0 (o índice em frames), ids a
partir de 1, caixa (x, y, w, h) amodal — a caixa da elipse inteira, mesmo
quando parcialmente coberta, como no gt.txt do MOT17 — e visibility = pixels
visíveis / pixels da elipse.
"""

import numpy as np

# Folga, em pixels, entre o alvo e a borda do oclusor na direção do
# movimento. Pequena para o buraco medido ficar perto de occlusion_frames.
OCCLUDER_SLACK = 0.5


def _rasterize_ellipse(yy, xx, cy, cx, a, b, theta):
    """
    Máscara booleana da elipse de centro (cy, cx), semieixo a na direção
    theta e b na perpendicular. Mesma rasterização do PA1.
    """

    ct, st = np.cos(theta), np.sin(theta)

    dy, dx = yy - cy, xx - cx

    u = (dx * ct + dy * st) / a
    v = (-dx * st + dy * ct) / b

    return (u * u + v * v) <= 1.0


def _half_extent(a, b, theta):
    """
    Meia largura e meia altura da caixa que envolve a elipse rotacionada.
    """

    ct, st = np.cos(theta), np.sin(theta)

    half_w = np.sqrt((a * ct) ** 2 + (b * st) ** 2)
    half_h = np.sqrt((a * st) ** 2 + (b * ct) ** 2)

    return float(half_w), float(half_h)


def _fold(p, lo, hi):
    """
    Reflete a coordenada p para dentro de [lo, hi]. Aplicado a uma trajetória
    retilínea, é a mesma trajetória quicando nas paredes.
    """

    span = hi - lo

    if span <= 0:
        return (lo + hi) / 2.0

    q = (p - lo) % (2.0 * span)

    return lo + (q if q <= span else 2.0 * span - q)


def _random_velocity(rng, speed):
    angle = rng.uniform(0.0, 2.0 * np.pi)
    magnitude = speed * rng.uniform(0.5, 1.5)

    return magnitude * np.array([np.cos(angle), np.sin(angle)])


def _fold_path(start, velocity, n_frames, lo, hi):
    """
    Centros por quadro de um movimento retilíneo que quica nas paredes.
    """

    centers = []

    for t in range(n_frames):
        p = start + velocity * t
        centers.append(np.array([_fold(p[0], lo[0], hi[0]),
                                 _fold(p[1], lo[1], hi[1])]))

    return centers


def _free_object(rng, size, speed, n_frames, a=None, b=None, theta=None):
    a = rng.uniform(3.0, 10.0) if a is None else a
    b = rng.uniform(3.0, 10.0) if b is None else b
    theta = rng.uniform(0.0, np.pi) if theta is None else theta

    lo = np.array(_half_extent(a, b, theta))
    hi = size - lo

    start = rng.uniform(lo, hi)
    velocity = _random_velocity(rng, speed)

    centers = _fold_path(start, velocity, n_frames, lo, hi)

    return {"a": a, "b": b, "theta": theta, "centers": centers}


def _occlusion_pair(rng, size, speed, n_frames, occlusion_frames):
    """
    Oclusor e alvo. O alvo tem o semieixo maior na direção do movimento
    relativo, e o oclusor tem nessa mesma direção o semieixo
    a_alvo + OCCLUDER_SLACK: assim o alvo fica inteiro coberto exatamente
    enquanto o deslocamento entre os centros for <= OCCLUDER_SLACK.
    """

    target_a = rng.uniform(4.0, 7.0)
    target_b = target_a * rng.uniform(0.5, 0.9)

    occluder_a = target_a + OCCLUDER_SLACK
    occluder_b = occluder_a * rng.uniform(1.0, 1.5)

    theta = rng.uniform(0.0, np.pi)
    direction = np.array([np.cos(theta), np.sin(theta)])

    occluder = _free_object(rng, size, speed, n_frames,
                            a=occluder_a, b=occluder_b, theta=theta)
    centers = occluder["centers"]

    t0 = (n_frames - occlusion_frames) // 2
    t1 = t0 + occlusion_frames - 1

    # Velocidade do oclusor já refletida, medida na borda da janela: é em
    # relação a ela que o alvo entra e sai de trás dele.
    v_before = centers[t0] - centers[t0 - 1] if t0 > 0 else 0.0
    v_after = centers[t1 + 1] - centers[t1] if t1 + 1 < n_frames else 0.0

    v_in = v_before + speed * rng.choice([-1.0, 1.0]) * direction
    v_out = v_after + speed * rng.choice([-1.0, 1.0]) * direction

    # Na janela o alvo copia o centro (já refletido) do oclusor. Fora dela
    # anda reto a partir das pontas da janela e quica nos próprios limites;
    # como o centro do oclusor está dentro dos limites do alvo (o alvo é
    # menor), a trajetória é contínua nas pontas.
    lo = np.array(_half_extent(target_a, target_b, theta))
    hi = size - lo

    before = _fold_path(centers[t0], -v_in, t0 + 1, lo, hi)[::-1]
    after = _fold_path(centers[t1], v_out, n_frames - t1, lo, hi)

    target = {
        "a": target_a,
        "b": target_b,
        "theta": theta,
        "centers": before[:-1] + centers[t0:t1 + 1] + after[1:],
    }

    return occluder, target


def depth_order(objects):
    """
    Ordem de desenho (fundo -> frente). Quem está na frente oclui quem está
    atrás; é isso que torna a oclusão verificável.
    """

    return sorted(range(len(objects)), key=lambda i: objects[i]["depth"])


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

    if occlusion_frames > 0:
        if speed <= 0:
            raise ValueError("oclusão exige speed > 0: o alvo precisa entrar "
                             "e sair de trás do oclusor")
        if n_objects < 2:
            raise ValueError("oclusão exige n_objects >= 2 (oclusor + alvo)")
        if n_frames - occlusion_frames < 2:
            raise ValueError("occlusion_frames precisa deixar pelo menos um "
                             "quadro visível antes e depois")

    rng = np.random.default_rng(seed)

    objects = []

    if occlusion_frames > 0:
        occluder, target = _occlusion_pair(rng, size, speed, n_frames,
                                           occlusion_frames)
        occluder["depth"] = 2.0
        target["depth"] = 1.5
        objects += [occluder, target]

    while len(objects) < n_objects:
        obj = _free_object(rng, size, speed, n_frames)
        obj["depth"] = rng.uniform(0.0, 1.0)
        objects.append(obj)

    background = rng.uniform(0.1, 0.3)

    for obj in objects:
        obj["intensity"] = background + contrast * rng.uniform(0.3, 0.7)

    # Ids embaralhados: o par da oclusão não fica sempre com os ids 1 e 2.
    ids = rng.permutation(n_objects) + 1

    order = depth_order(objects)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    frames = np.empty((n_frames, size, size), np.float32)
    tracks = []

    for t in range(n_frames):

        # Mapa de quem é dono de cada pixel. O mesmo mapa pinta a imagem e
        # mede a visibilidade, então pixel e ground truth não divergem.
        owner = np.full((size, size), -1, np.int32)
        masks = {}

        for i in order:
            obj = objects[i]
            cx, cy = obj["centers"][t]

            mask = _rasterize_ellipse(yy, xx, cy, cx,
                                      obj["a"], obj["b"], obj["theta"])
            masks[i] = mask
            owner[mask] = i

        image = np.full((size, size), background, np.float32)

        for i, obj in enumerate(objects):
            image[owner == i] = obj["intensity"]

        if noise > 0:
            image = image + rng.normal(0.0, noise, image.shape)

        frames[t] = np.clip(image, 0.0, 1.0)

        for i, obj in enumerate(objects):
            full = int(masks[i].sum())
            visible = int((owner == i).sum())

            if full == 0 or visible == 0:
                continue

            half_w, half_h = _half_extent(obj["a"], obj["b"], obj["theta"])
            cx, cy = obj["centers"][t]

            tracks.append((
                t,
                int(ids[i]),
                cx - half_w,
                cy - half_h,
                2.0 * half_w,
                2.0 * half_h,
                visible / full,
            ))

    tracks.sort(key=lambda row: (row[0], row[1]))

    return frames, tracks


def occlusion_durations(tracks):
    """
    Duração de cada buraco de visibilidade por identidade. Usado na Parte 4
    para comparar o horizonte de memória com a distribuição do dataset.

    Só conta buracos internos (a identidade aparece antes e depois).
    Devolve {id: [duração do 1º buraco, do 2º, ...]}.
    """

    frames_by_id = {}

    for row in tracks:
        frames_by_id.setdefault(row[1], []).append(row[0])

    durations = {}

    for track_id, frames in frames_by_id.items():
        frames = sorted(frames)
        durations[track_id] = [b - a - 1 for a, b in zip(frames, frames[1:])
                               if b - a > 1]

    return durations
