"""Olhe as aumentações com seus próprios olhos: o animal (às vezes <2% do quadro) sobrevive ao recorte?

  python -m tests.viz_aug --data dados --out resultados/diagnosticos/aug.png [--set aug.weak_scale_min=0.5 ...]
Cada linha: imagem como o modelo vê (eval) | 3 vistas fracas | 3 vistas fortes (FixMatch).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import common as C  # noqa: E402
from src import fixmatch as FM  # noqa: E402
from src import preproc  # noqa: E402


def denorm(t):
    a = t.numpy().transpose(1, 2, 0) * np.array(preproc.STD) + np.array(preproc.MEAN)
    return np.clip(a, 0, 1)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="resultados/diagnosticos/aug.png")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], metavar="a.b=v")
    a = ap.parse_args()
    cfg = C.method_cfg("fixmatch", a.budget, a.set, allow_defaults=True)
    data = C.load_data(a.data, a.budget, cfg["data"]["size"], cfg["data"]["resize_mode"], cfg["data"]["preload_factor"],
                       need_unl=True)
    C.seed_all(a.seed)
    ds = FM.UnlDS(data.unl, data.size, cfg["aug"])
    idx = np.random.RandomState(a.seed).choice(len(data.unl), a.n, replace=False)
    fig, axs = plt.subplots(a.n, 7, figsize=(14, 2.1 * a.n))
    for r, i in enumerate(idx):
        axs[r, 0].imshow(C.Image.fromarray(data.unl[i]).resize((data.size, data.size)))
        axs[r, 0].set_title("entrada", fontsize=7)
        for k in range(3):
            w, s, _ = ds[int(i)]
            axs[r, 1 + k].imshow(denorm(w))
            axs[r, 4 + k].imshow(denorm(s))
            axs[r, 1 + k].set_title("fraca", fontsize=7)
            axs[r, 4 + k].set_title("forte", fontsize=7)
        for ax in axs[r]:
            ax.axis("off")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("salvo em", a.out)


if __name__ == "__main__":
    main()
