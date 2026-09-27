"""
Regimes de treino do modelo temporal (Eixo 2 da Parte 3).

Na inferência o modelo se alimenta das próprias previsões — e, sob oclusão, só
delas. Se nunca viu isso no treino, a distribuição muda debaixo dele. Os três
regimes existem para medir exatamente esse descolamento.
"""

import torch

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


def truncated_bptt(model, window, loss_fn, regime="teacher_forcing",
                   clip_grad=None, optimizer=None):
    """
    Um passo de BPTT truncado sobre uma janela de T quadros.

    `window` é um dicionário {id: [(frame, embedding), ...]} com embeddings
    já L2 (tensores 1-D). Teacher forcing na Trilha B: sempre a observação
    verdadeira. `clip_grad=None` desliga o clipping.
    """

    del regime
    device = next(model.parameters()).device
    frames = sorted({frame for steps in window.values() for frame, _ in steps})
    states = {}
    outputs = []

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

        if model.cell == "lstm":
            state = (torch.cat([item[0] for item in packed], dim=1),
                     torch.cat([item[1] for item in packed], dim=1))
        else:
            state = torch.cat(packed, dim=1)

        vectors, new_state = model.step(batch, state)

        for i, identity in enumerate(identities):
            if model.cell == "lstm":
                states[identity] = (new_state[0][:, i:i + 1],
                                    new_state[1][:, i:i + 1])
            else:
                states[identity] = new_state[:, i:i + 1]
            outputs.append((identity, vectors[i]))

    # InfoNCE/triplet precisam de dois vetores do mesmo id. Por quadro
    # cada id aparece uma vez; a perda olha a janela inteira.
    if len(outputs) < 2:
        return torch.zeros((), device=device, requires_grad=True)

    embeddings = torch.stack([vector for _, vector in outputs])
    labels = torch.tensor([identity for identity, _ in outputs],
                          device=device)
    loss = loss_fn(embeddings, labels)

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

    return loss


def gradient_norm_profile(model, window):
    """
    Norma de dL_t/dh_{t-k} em função de k — a curva de gradiente que some, na
    forma analítica que a Parte 4 exige.
    """

    raise NotImplementedError
