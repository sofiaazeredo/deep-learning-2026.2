"""
Regimes de treino do modelo temporal (Eixo 2 da Parte 3).

Na inferência o modelo se alimenta das próprias previsões — e, sob oclusão, só
delas. Se nunca viu isso no treino, a distribuição muda debaixo dele. Os três
regimes existem para medir exatamente esse descolamento.
"""

import torch

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
    """

    del regime
    kind = getattr(model, "kind", "appearance")
    device = next(model.parameters()).device
    frames = sorted({frame for steps in window.values() for frame, _ in steps})
    states = {}
    outputs = []
    pending_box = {}
    box_losses = []

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

        for i, identity in enumerate(identities):
            states[identity] = _split_state(new_state, i, model.cell)
            if vectors is not None:
                outputs.append((identity, vectors[i]))
            if boxes is not None and box_loss_fn is not None:
                current_box = batch[i, -4:] if kind == "both" else batch[i]
                if identity in pending_box:
                    box_losses.append(box_loss_fn(pending_box[identity],
                                                  current_box))
                pending_box[identity] = boxes[i]

    losses = []
    if kind in {"appearance", "both"} and loss_fn is not None:
        if len(outputs) < 2:
            losses.append(torch.zeros((), device=device, requires_grad=True))
        else:
            embeddings = torch.stack([vector for _, vector in outputs])
            labels = torch.tensor([identity for identity, _ in outputs],
                                  device=device)
            losses.append(loss_fn(embeddings, labels))

    if box_losses:
        losses.append(torch.stack(box_losses).mean())

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
        vectors, _ = unpack_step(raw, getattr(model, "kind", "appearance"))
        for i, identity in enumerate(identities):
            hid = _split_state(new_state, i, model.cell)
            hidden = hid[0] if model.cell == "lstm" else hid
            hidden.retain_grad()
            hidden_steps.append((t_idx, hidden))
            states[identity] = hid
            if vectors is not None:
                outputs_by_frame.setdefault(frame, []).append(
                    (identity, vectors[i]))

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
