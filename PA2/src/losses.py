"""
Perdas do modelo temporal (Parte 2).

Trilha A: smooth-L1 sobre a caixa prevista; opcionalmente log-verossimilhança
gaussiana quando o modelo também prevê a incerteza.

Trilha B: contrastiva ou triplet sobre as identidades do ground truth.
"""

import torch
import torch.nn.functional as F


class SmoothL1BoxLoss:
    def __init__(self, beta=1.0, parameterization="cxcywh"):
        raise NotImplementedError


class GaussianNLLBoxLoss:
    """
    -log N(caixa_verdadeira | mu, sigma). O sigma aprendido vira o portão
    adaptativo em src.association.gate.
    """

    def __init__(self, min_sigma=1e-3):
        raise NotImplementedError


class TripletIdentityLoss:
    def __init__(self, margin=0.3):
        self.margin = float(margin)

    def __call__(self, embeddings, labels):
        if embeddings.shape[0] < 2:
            return embeddings.sum() * 0.0

        labels = labels.view(-1)
        distances = torch.cdist(embeddings, embeddings, p=2)
        loss = embeddings.new_zeros(())
        n_valid = 0

        for i, label in enumerate(labels):
            pos = (labels == label) & (torch.arange(len(labels),
                                                    device=labels.device) != i)
            neg = labels != label
            if not pos.any() or not neg.any():
                continue
            hardest_pos = distances[i][pos].max()
            hardest_neg = distances[i][neg].min()
            loss = loss + F.relu(hardest_pos - hardest_neg + self.margin)
            n_valid += 1

        return loss / n_valid if n_valid else embeddings.sum() * 0.0


class ContrastiveIdentityLoss:
    def __init__(self, temperature=0.07):
        self.temperature = float(temperature)

    def __call__(self, embeddings, labels):
        """
        InfoNCE: para cada vetor, os de mesmo id são positivos contra o resto
        do lote. Sem positivo no lote a linha é ignorada.
        """

        n = embeddings.shape[0]
        if n < 2:
            return embeddings.sum() * 0.0

        labels = labels.view(-1)
        logits = embeddings @ embeddings.T / self.temperature
        self_mask = torch.eye(n, dtype=torch.bool, device=embeddings.device)
        logits = logits.masked_fill(self_mask, float("-inf"))

        positive = labels.unsqueeze(0) == labels.unsqueeze(1)
        positive = positive & ~self_mask
        valid = positive.any(dim=1)
        if not valid.any():
            return embeddings.sum() * 0.0

        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        n_pos = positive.sum(dim=1).clamp(min=1)
        per_row = (log_prob.masked_fill(~positive, 0.0).sum(dim=1) / n_pos)
        return -per_row[valid].mean()
