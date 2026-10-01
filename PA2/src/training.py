"""
Regimes de treino do modelo temporal (Eixo 2 da Parte 3).

Na inferência o modelo se alimenta das próprias previsões — e, sob oclusão, só
delas. Se nunca viu isso no treino, a distribuição muda debaixo dele. Os três
regimes existem para medir exatamente esse descolamento.
"""

import torch

from src.boxes import encode_delta
from src.model import unpack_step

REGIMES = ("teacher_forcing", "scheduled_sampling", "free_running")


def sampling_probability(regime, epoch, total_epochs):
    """
    Probabilidade de alimentar a observação verdadeira no passo seguinte.
    Constante 1 no teacher forcing, decaindo no scheduled sampling, 0 no
    free-running.
    """

    if regime == "teacher_forcing":
        return 1.0
    if regime == "free_running":
        return 0.0
    if regime == "scheduled_sampling":
        if total_epochs <= 1:
            return 0.0
        return max(0.0, 1.0 - epoch / (total_epochs - 1))
    raise ValueError(f"regime desconhecido: {regime}")


def _cat_states(packed, cell):
    if cell == "lstm":
        return (torch.cat([item[0] for item in packed], dim=1),
                torch.cat([item[1] for item in packed], dim=1))
    return torch.cat(packed, dim=1)


def _split_state(new_state, index, cell):
    if cell == "lstm":
        return (new_state[0][:, index:index + 1],
                new_state[1][:, index:index + 1])
    return new_state[:, index:index + 1]


def _stack(items, device):
    """
    [(identity, frame, vector)] -> (vetores, ids, quadros), para a perda
    preditiva.
    """

    vectors = torch.stack([vector for _, _, vector in items])
    labels = torch.tensor([identity for identity, _, _ in items], device=device)
    frames = torch.tensor([frame for _, frame, _ in items], device=device)
    return vectors, labels, frames


def truncated_bptt(model, window, loss_fn=None, regime="teacher_forcing",
                   clip_grad=None, optimizer=None, box_loss_fn=None):
    """
    Um passo de BPTT truncado sobre uma janela de T quadros.

    `window` é {id: [(frame, features), ...]} com tensores 1-D:
      aparência  (D,)
      geometria  (4,)
      both       (D+4,)
    Teacher forcing: sempre a observação verdadeira. `clip_grad=None`
    desliga o clipping.

    Com uma perda preditiva (`loss_fn.predictive`), cada observação gera a
    memória da track depois dela e a consulta do recorte (cabeça sem
    estado); a perda casa memória com o futuro da mesma identidade.
    """

    del regime
    kind = getattr(model, "kind", "appearance")
    device = next(model.parameters()).device
    frames = sorted({frame for steps in window.values() for frame, _ in steps})
    states = {}
    outputs = []
    queries = []
    pending_box = {}
    box_losses = []
    predictive = getattr(loss_fn, "predictive", False)

    model.train()

    for frame in frames:
        present = [(identity, embedding.to(device))
                   for identity, steps in window.items()
                   for step_frame, embedding in steps
                   if step_frame == frame]
        if not present:
            continue

        identities = [identity for identity, _ in present]
        batch = torch.stack([embedding for _, embedding in present])
        packed = [states.get(identity, model.init_state(1, device))
                  for identity in identities]
        state = _cat_states(packed, model.cell)
        raw, new_state = model.step(batch, state)
        vectors, boxes = unpack_step(raw, kind)
        if predictive and vectors is not None:
            appearance = batch[:, :model.embed_dim] if kind == "both" else batch
            frame_queries = model.query(appearance)

        for i, identity in enumerate(identities):
            states[identity] = _split_state(new_state, i, model.cell)
            if vectors is not None:
                outputs.append((identity, frame, vectors[i]))
                if predictive:
                    queries.append((identity, frame, frame_queries[i]))
            if boxes is not None and box_loss_fn is not None:
                offset = model.embed_dim if kind == "both" else 0
                current_box = batch[i, offset:offset + 4]
                # A cabeça prevê o quadro SEGUINTE: só o par de quadros
                # consecutivos entra na perda (buraco de visibilidade não
                # vira um "passo" gigante). Perda no espaço codificado, em
                # volta da caixa de onde a previsão partiu.
                # Os pares são codificados de uma vez no fim da janela: um
                # encode por par eram ~40 kernels minúsculos por id e quadro.
                if identity in pending_box:
                    last_frame, origin, predicted = pending_box[identity]
                    if frame == last_frame + 1:
                        box_losses.append((origin, predicted, current_box))
                pending_box[identity] = (frame, current_box, boxes[i])

    losses = []
    if kind in {"appearance", "both"} and loss_fn is not None:
        if len(outputs) < 2:
            losses.append(torch.zeros((), device=device, requires_grad=True))
        elif predictive:
            losses.append(loss_fn(*_stack(outputs, device),
                                  *_stack(queries, device)))
        else:
            embeddings = torch.stack([vector for _, _, vector in outputs])
            labels = torch.tensor([identity for identity, _, _ in outputs],
                                  device=device)
            losses.append(loss_fn(embeddings, labels))

    if box_losses:
        origin, predicted, target = (torch.stack(part) for part in zip(*box_losses))
        losses.append(box_loss_fn(encode_delta(origin, predicted),
                                  encode_delta(origin, target)))

    if not losses:
        return torch.zeros((), device=device, requires_grad=True)

    loss = torch.stack(losses).mean()

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

    return loss


def gradient_norm_profile(model, window, loss_fn=None):
    """
    Norma de dL_t/dh_{t-k} em função de k — a curva de gradiente que some, na
    forma analítica que a Parte 4 exige.

    `window` é o mesmo dicionário do BPTT. L_t é a InfoNCE no último quadro
    com pelo menos dois ids; devolve [||dL/dh_t||, ||dL/dh_{t-1}||, ...].

    Com a perda preditiva, L_t é a perda das memórias do último quadro t que
    ainda tem futuro na janela, contra as consultas (que não dependem de h).
    O único caminho de L_t até h_{t-k} continua sendo a recorrência.
    """

    from src.losses import ContrastiveIdentityLoss

    if loss_fn is None:
        loss_fn = ContrastiveIdentityLoss()

    device = next(model.parameters()).device
    model.zero_grad(set_to_none=True)
    model.train()

    frames = sorted({frame for steps in window.values() for frame, _ in steps})
    states = {}
    hidden_steps = []
    outputs_by_frame = {}
    queries = []
    predictive = getattr(loss_fn, "predictive", False)

    for t_idx, frame in enumerate(frames):
        present = [(identity, embedding.to(device))
                   for identity, steps in window.items()
                   for step_frame, embedding in steps
                   if step_frame == frame]
        if not present:
            continue
        identities = [identity for identity, _ in present]
        batch = torch.stack([embedding for _, embedding in present])
        packed = [states.get(identity, model.init_state(1, device))
                  for identity in identities]
        state = _cat_states(packed, model.cell)
        raw, new_state = model.step(batch, state)
        kind = getattr(model, "kind", "appearance")
        vectors, _ = unpack_step(raw, kind)
        if predictive:
            appearance = batch[:, :model.embed_dim] if kind == "both" else batch
            frame_queries = model.query(appearance).detach()
            for i, identity in enumerate(identities):
                queries.append((identity, frame, frame_queries[i]))
        for i, identity in enumerate(identities):
            hid = _split_state(new_state, i, model.cell)
            hidden = hid[0] if model.cell == "lstm" else hid
            hidden.retain_grad()
            hidden_steps.append((t_idx, hidden))
            states[identity] = hid
            if vectors is not None:
                outputs_by_frame.setdefault(frame, []).append(
                    (identity, vectors[i]))

    if predictive:
        return _predictive_profile(frames, outputs_by_frame, queries,
                                   hidden_steps, loss_fn, device)

    # L_t só no último quadro que ainda tem positivo no passado: assim o
    # único caminho até h_{t-k} é a recorrência (a curva que some).
    last_t = None
    last_items = hist_items = None
    for frame in reversed(frames):
        items = outputs_by_frame.get(frame, [])
        history = [(identity, vector)
                   for older, rows in outputs_by_frame.items() if older < frame
                   for identity, vector in rows]
        now = {identity for identity, _ in items}
        past = {identity for identity, _ in history}
        if items and now & past:
            last_items, hist_items = items, history
            last_t = next(t_idx for t_idx, step_frame in enumerate(frames)
                          if step_frame == frame)
            break
    if last_items is None:
        return []

    embeddings = torch.cat([
        torch.stack([vector for _, vector in last_items]),
        torch.stack([vector.detach() for _, vector in hist_items]),
    ])
    labels = torch.tensor(
        [identity for identity, _ in last_items]
        + [identity for identity, _ in hist_items],
        device=device)
    loss = loss_fn(embeddings, labels)
    if float(loss.detach()) == 0:
        return []
    loss.backward()
    by_k = {}
    for t_idx, hidden in hidden_steps:
        if hidden.grad is None or t_idx > last_t:
            continue
        by_k.setdefault(last_t - t_idx, []).append(
            float(hidden.grad.detach().norm()))

    return [float(sum(by_k[k]) / len(by_k[k])) for k in sorted(by_k)]


def _predictive_profile(frames, outputs_by_frame, queries, hidden_steps,
                        loss_fn, device):
    future_of = {}
    for identity, frame, _ in queries:
        future_of.setdefault(identity, []).append(frame)

    last_t = None
    for t_idx in range(len(frames) - 1, -1, -1):
        frame = frames[t_idx]
        items = outputs_by_frame.get(frame, [])
        if any(max(future_of.get(identity, [frame])) > frame
               for identity, _ in items):
            last_t, last_items, last_frame = t_idx, items, frame
            break
    if last_t is None:
        return []

    memory = torch.stack([vector for _, vector in last_items])
    mem_labels = torch.tensor([identity for identity, _ in last_items],
                              device=device)
    mem_frames = torch.full((len(last_items),), last_frame, device=device)
    loss = loss_fn(memory, mem_labels, mem_frames, *_stack(queries, device))
    if float(loss.detach()) == 0:
        return []
    loss.backward()

    by_k = {}
    for t_idx, hidden in hidden_steps:
        if hidden.grad is None or t_idx > last_t:
            continue
        by_k.setdefault(last_t - t_idx, []).append(
            float(hidden.grad.detach().norm()))

    return [float(sum(by_k[k]) / len(by_k[k])) for k in sorted(by_k)]
