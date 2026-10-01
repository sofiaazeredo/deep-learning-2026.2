"""
Modelo temporal (Parte 2) e as células comparadas no Eixo 1 da Parte 3.

Trilha A — RNN como modelo de movimento: o estado recorrente recebe a última
observação (caixa, opcionalmente confiança e dt) e prevê a caixa do quadro
seguinte. Sob oclusão roda para frente sem observação.

Trilha B — RNN como memória de aparência: um agregador recorrente mantém o
estado de aparência da track, atualizado a cada observação.

A escolha da trilha é única e fica registrada no README. O Eixo 3 da Parte 3
varia o que entra na recorrência (geometria / aparência / os dois) sem
trocar a trilha da Parte 2.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CELLS = ("rnn", "lstm", "gru")
_CELL = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
KIND_TO_NAME = {"appearance": "appearance", "geometry": "motion",
                "both": "fusion"}
# "delta": a cabeça de geometria prevê o deslocamento codificado
# (src.boxes.encode_delta) a partir da caixa atual + velocidade (8 de
# entrada). "absolute": a versão antiga, que regredia a caixa normalizada
# direto (4 de entrada) e perdia para "copiar a última caixa" — só para
# carregar os checkpoints velhos.
PARAMETERIZATIONS = ("delta", "absolute")


def _box_feature_dim(parameterization):
    if parameterization not in PARAMETERIZATIONS:
        raise ValueError(f"parametrização desconhecida: {parameterization}")
    return 8 if parameterization == "delta" else 4


def _decode_box(current, raw, parameterization):
    from src.boxes import decode_delta

    if parameterization == "delta":
        return decode_delta(current, raw)
    return raw


def _init_state(cell, hidden, batch, device):
    state = torch.zeros(1, batch, hidden, device=device)
    if cell == "lstm":
        return (state, torch.zeros_like(state))
    return state


def _split_state(new_state, index, cell):
    if cell == "lstm":
        return (new_state[0][:, index:index + 1],
                new_state[1][:, index:index + 1])
    return new_state[:, index:index + 1]


class MotionRNN(nn.Module):
    """
    Cabeça de geometria do Eixo 3. Recebe a caixa atual + a velocidade
    codificada (src.boxes.motion_features) e prevê a caixa do quadro
    seguinte: a saída é o deslocamento codificado, e `step` já devolve a
    caixa absoluta. `predict_sigma` e `bidirectional` ficam para outros
    eixos.
    """

    kind = "geometry"

    def __init__(self, cell="gru", hidden=128, box_dim=4, predict_sigma=False,
                 bidirectional=False, parameterization="delta"):
        super().__init__()
        if cell not in _CELL:
            raise ValueError(f"célula desconhecida: {cell}")
        if predict_sigma:
            raise ValueError("predict_sigma é o portão adaptativo da Trilha A")
        if bidirectional:
            raise ValueError("bidirecional é o Eixo 4 da Parte 3")

        self.cell = cell
        self.hidden = int(hidden)
        self.box_dim = int(box_dim)
        self.parameterization = parameterization
        self.feature_dim = _box_feature_dim(parameterization)
        self.rnn = _CELL[cell](self.feature_dim, self.hidden, batch_first=True)
        self.proj = nn.Linear(self.hidden, self.box_dim)

    def init_state(self, batch=1, device="cpu"):
        return _init_state(self.cell, self.hidden, batch, device)

    def step(self, features, state):
        if features.dim() == 1:
            features = features.unsqueeze(0)
        output, state = self.rnn(features.unsqueeze(1), state)
        raw = self.proj(output.squeeze(1))
        return _decode_box(features[:, :4], raw, self.parameterization), state


class AppearanceRNN(nn.Module):
    """
    Trilha B. Agrega embeddings de recortes ao longo do tempo num estado de
    aparência por track (a memória, saída de `step`).

    A detecção nova entra na associação pela `query`, uma cabeça própria e
    sem estado. Memória e consulta vivem no mesmo espaço porque a perda
    preditiva treina uma contra a outra — não por construção.
    """

    kind = "appearance"
    has_query = True

    def __init__(self, cell="gru", embed_dim=128, hidden=128):
        super().__init__()
        if cell not in _CELL:
            raise ValueError(f"célula desconhecida: {cell}")

        self.cell = cell
        self.embed_dim = int(embed_dim)
        self.hidden = int(hidden)
        self.rnn = _CELL[cell](self.embed_dim, self.hidden, batch_first=True)
        self.proj = nn.Linear(self.hidden, self.embed_dim)
        self.query_head = nn.Linear(self.embed_dim, self.embed_dim)

    def init_state(self, batch=1, device="cpu"):
        return _init_state(self.cell, self.hidden, batch, device)

    def query(self, embedding):
        """
        Consulta de um recorte: embedding (B, D) -> vetor L2 (B, D).
        """

        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        return F.normalize(self.query_head(embedding), dim=-1)

    def step(self, embedding, state):
        """
        Um passo: embedding (B, D) e estado -> (vetor L2, novo estado).
        """

        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        output, state = self.rnn(embedding.unsqueeze(1), state)
        vector = F.normalize(self.proj(output.squeeze(1)), dim=-1)
        return vector, state


class FusionRNN(nn.Module):
    """
    Eixo 3, braço 'both': embedding e caixa no mesmo estado recorrente.
    """

    kind = "both"
    has_query = True

    def __init__(self, cell="gru", embed_dim=128, hidden=128, box_dim=4,
                 parameterization="delta"):
        super().__init__()
        if cell not in _CELL:
            raise ValueError(f"célula desconhecida: {cell}")

        self.cell = cell
        self.embed_dim = int(embed_dim)
        self.hidden = int(hidden)
        self.box_dim = int(box_dim)
        self.parameterization = parameterization
        self.feature_dim = self.embed_dim + _box_feature_dim(parameterization)
        self.rnn = _CELL[cell](self.feature_dim, self.hidden,
                               batch_first=True)
        self.proj_emb = nn.Linear(self.hidden, self.embed_dim)
        self.proj_box = nn.Linear(self.hidden, self.box_dim)
        self.query_head = nn.Linear(self.embed_dim, self.embed_dim)

    def init_state(self, batch=1, device="cpu"):
        return _init_state(self.cell, self.hidden, batch, device)

    def query(self, embedding):
        """
        Consulta de um recorte: só a parte de aparência (B, D), sem a caixa.
        """

        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        return F.normalize(self.query_head(embedding), dim=-1)

    def step(self, features, state):
        if features.dim() == 1:
            features = features.unsqueeze(0)
        output, state = self.rnn(features.unsqueeze(1), state)
        hidden = output.squeeze(1)
        vector = F.normalize(self.proj_emb(hidden), dim=-1)
        current = features[:, self.embed_dim:self.embed_dim + 4]
        box = _decode_box(current, self.proj_box(hidden), self.parameterization)
        return (vector, box), state


def unpack_step(output, kind="appearance"):
    if kind == "both":
        return output[0], output[1]
    if kind == "geometry":
        return None, output
    return output, None


def build_model(name, **kwargs):
    """
    Registro nome -> modelo, e loader de checkpoint (mesmo padrão do PA1).
    """

    if name in {"appearance", "b", "track_b"}:
        allowed = {key: kwargs[key] for key in ("cell", "embed_dim", "hidden")
                   if key in kwargs}
        return AppearanceRNN(**allowed)
    if name in {"motion", "geometry"}:
        allowed = {key: kwargs[key]
                   for key in ("cell", "hidden", "box_dim", "parameterization")
                   if key in kwargs}
        return MotionRNN(**allowed)
    if name in {"fusion", "both"}:
        allowed = {key: kwargs[key]
                   for key in ("cell", "embed_dim", "hidden", "box_dim",
                               "parameterization")
                   if key in kwargs}
        return FusionRNN(**allowed)
    raise ValueError(f"modelo desconhecido: {name}")


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    kwargs = dict(payload["kwargs"])
    if (payload.get("name") in {"motion", "geometry", "fusion", "both"}
            and "parameterization" not in kwargs):
        # Checkpoint anterior à cabeça de deslocamento.
        kwargs["parameterization"] = "absolute"
    model = build_model(payload.get("name", "appearance"), **kwargs)
    missing, unexpected = model.load_state_dict(payload["state_dict"],
                                                strict=False)
    if unexpected or any(not key.startswith("query_head") for key in missing):
        raise RuntimeError(f"checkpoint incompatível: faltam {missing}, "
                           f"sobram {unexpected}")
    if missing:
        # Checkpoint anterior à cabeça de consulta: o tracker usa a consulta
        # antiga (um passo da GRU a partir do estado zero).
        model.has_query = False
    model.to(device)
    model.eval()
    return model, payload


def save_checkpoint(path, model, kwargs, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(kwargs)
    if hasattr(model, "parameterization"):
        # Gravado sempre: sem isso o loader trataria o checkpoint como da
        # cabeça antiga (absoluta).
        kwargs["parameterization"] = model.parameterization
    name = KIND_TO_NAME.get(getattr(model, "kind", "appearance"), "appearance")
    if extra and extra.get("name"):
        name = extra["name"]
    payload = {"name": name, "kwargs": kwargs,
               "state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
        payload["name"] = name
    torch.save(payload, path)
    return path
