"""Inferência sobre uma lista de arquivos de imagem. Usada por predict.py, galeria.py e pelos testes."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import preproc


class _PathDS(Dataset):
    def __init__(self, paths, size, mode):
        self.paths, self.size, self.mode = list(paths), size, mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return preproc.preprocess_path(self.paths[i], self.size, self.mode)


@torch.no_grad()
def predict_paths(model, paths, size, mode, device="cpu", bs=32, workers=0):
    """Retorna probabilidades softmax (N, K) como numpy."""
    model = model.to(device).eval()
    out = []
    for x in DataLoader(_PathDS(paths, size, mode), batch_size=bs, num_workers=workers):
        out.append(model(x.to(device)).softmax(1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 0), dtype=np.float32)
