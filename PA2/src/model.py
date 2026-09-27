"""
Modelo temporal (Parte 2) e as células comparadas no Eixo 1 da Parte 3.

Trilha A — RNN como modelo de movimento: o estado recorrente recebe a última
observação (caixa, opcionalmente confiança e dt) e prevê a caixa do quadro
seguinte. Sob oclusão roda para frente sem observação.

Trilha B — RNN como memória de aparência: um agregador recorrente mantém o
estado de aparência da track, atualizado a cada observação.

A escolha da trilha é única e fica registrada no README.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CELLS = ("rnn", "lstm", "gru")
_CELL = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}


class MotionRNN:
    """
    Trilha A. `cell` em CELLS, com orçamento de parâmetros aproximadamente
    igual entre as células (Eixo 1). `predict_sigma=True` emite também a
    incerteza, que vira portão de associação adaptativo.

    `bidirectional=True` é o modo offline do Eixo 4 — não pode ser usado na
    avaliação online.
    """

    def __init__(self, cell="gru", hidden=128, predict_sigma=False,
                 bidirectional=False):
        raise NotImplementedError


class AppearanceRNN(nn.Module):
    """
    Trilha B. Agrega embeddings de recortes ao longo do tempo num estado de
    aparência por track.
    """

    def __init__(self, cell="gru", embed_dim=128, hidden=128):
        super().__init__()
        if cell not in _CELL:
            raise ValueError(f"célula desconhecida: {cell}")

        self.cell = cell
        self.embed_dim = int(embed_dim)
        self.hidden = int(hidden)
        self.rnn = _CELL[cell](self.embed_dim, self.hidden, batch_first=True)
        self.proj = nn.Linear(self.hidden, self.embed_dim)

    def init_state(self, batch=1, device="cpu"):
        hidden = torch.zeros(1, batch, self.hidden, device=device)
        if self.cell == "lstm":
            return (hidden, torch.zeros_like(hidden))
        return hidden

    def step(self, embedding, state):
        """
        Um passo: embedding (B, D) e estado -> (vetor L2, novo estado).
        """

        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        output, state = self.rnn(embedding.unsqueeze(1), state)
        vector = F.normalize(self.proj(output.squeeze(1)), dim=-1)
        return vector, state


def build_model(name, **kwargs):
    """
    Registro nome -> modelo, e loader de checkpoint (mesmo padrão do PA1).
    """

    if name in {"appearance", "b", "track_b"}:
        return AppearanceRNN(**kwargs)
    raise ValueError(f"modelo desconhecido: {name}")


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(payload.get("name", "appearance"), **payload["kwargs"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def save_checkpoint(path, model, kwargs, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"name": "appearance", "kwargs": kwargs,
               "state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path
