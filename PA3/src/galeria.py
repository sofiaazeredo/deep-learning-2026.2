"""Parte 3 — galeria de falhas e atalhos do MODELO FINAL.

 * N erros: imagem, classe verdadeira, prevista, confiança, Grad-CAM e oclusão + números do mapa.
 * N acertos (sorteados com seed fixa, sem escolher a dedo): o modelo olhou para o animal ou para o fundo?
O test_public é usado só para reportar; nada aqui altera o modelo. O diagnóstico em texto é de vocês
(a coluna `diagnostico` fica vazia nos CSVs). Para trocar os erros escolhidos: --ids ID1 ID2 ID3.

  python -m src.galeria --data dados --ckpt pesos/modelo_final.pt --out resultados/parte3
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import common as C
from . import saliencia as S
from .infer import predict_paths
from .models import load_checkpoint, target_layer


def image_attrs(path):
    """Atributos contínuos (sem limiares chutados): para vocês julgarem noite/tamanho nos diagnósticos."""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
    sat = float(np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean())
    return dict(largura=w, altura=h, cor_media_abs_diff=sat, brilho_medio=float(rgb.mean()))


def panel(ax_row, img, cam, occ, title, patch):
    for ax, im, t in zip(ax_row, [img, S.overlay(img, cam), S.overlay(img, occ)],
                         [title, "Grad-CAM", f"oclusão ({patch}px)"]):
        ax.imshow(im)
        ax.set_title(t, fontsize=8)
        ax.axis("off")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="pesos/modelo_final.pt")
    ap.add_argument("--out", default="resultados/parte3")
    ap.add_argument("--n_erros", type=int, default=3)
    ap.add_argument("--n_acertos", type=int, default=5)
    ap.add_argument("--criterio", choices=["conf_alta", "aleatorio"], default="conf_alta",
                    help="conf_alta: erros mais confiantes, uma classe verdadeira por erro; aleatorio: sorteio")
    ap.add_argument("--ids", nargs="*", default=None, help="força estes image_id como erros")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_candidatos", type=int, default=30)
    ap.add_argument("--cam_thr", type=float, default=0.5)
    ap.add_argument("--patch", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    dev = C.get_device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, ck = load_checkpoint(a.ckpt)
    size, mode = ck["preprocess"]["size"], ck["preprocess"]["resize_mode"]
    schema = C.load_schema()
    names, id2idx = C.load_classes(a.data, schema)
    assert names == ck["class_names"], "ordem/nomes de classes do checkpoint diferem de classes.csv"
    ids, paths, y = C.read_split(a.data, schema, "test_public", names, id2idx)
    probs = predict_paths(model, paths, size, mode, dev, bs=32)
    pred, conf = probs.argmax(1), probs.max(1)
    wrong = np.where(pred != y)[0]
    print(f"test_public: {len(y)} imagens | {len(wrong)} erros | acc {1 - len(wrong) / len(y):.3f}")

    cand = pd.DataFrame(dict(image_id=[ids[i] for i in wrong], verdadeira=[names[y[i]] for i in wrong],
                             prevista=[names[pred[i]] for i in wrong], confianca=conf[wrong], idx=wrong))
    cand = cand.sort_values("confianca", ascending=False).head(a.n_candidatos)
    cand = pd.concat([cand.reset_index(drop=True),
                      pd.DataFrame([image_attrs(paths[i]) for i in cand.idx])], axis=1)
    cand["diagnostico"] = ""
    cand.drop(columns="idx").to_csv(out / "candidatos_erros.csv", index=False)

    rng = np.random.RandomState(a.seed)
    if a.ids:
        err_idx = [ids.index(i) for i in a.ids]
    elif a.criterio == "aleatorio":
        err_idx = list(rng.choice(wrong, size=min(a.n_erros, len(wrong)), replace=False))
    else:
        err_idx, seen = [], set()
        for i in cand.idx:
            if y[i] not in seen:
                seen.add(y[i])
                err_idx.append(int(i))
            if len(err_idx) == a.n_erros:
                break
    right = np.where(pred == y)[0]
    hit_idx = list(rng.choice(right, size=min(a.n_acertos, len(right)), replace=False))

    gc = S.GradCAM(model.to(dev), target_layer(model, ck["arch"]))
    patch = a.patch or size // 8
    rows = []
    for kind, idxs in (("erro", err_idx), ("acerto", hit_idx)):
        fig, axs = plt.subplots(len(idxs), 3, figsize=(10, 3.4 * len(idxs)), squeeze=False)
        for r, i in enumerate(idxs):
            img, x = S.input_view(paths[i], size, mode)
            x = x.to(dev)
            cam, _, t = gc(x, int(pred[i]))
            with torch.no_grad():
                occ, drop = S.occlusion_map(model, x, t, patch, patch // 2)
            st = S.cam_stats(cam, a.cam_thr)
            title = (f"{kind}: {ids[i]}\nverdadeira={names[y[i]]} | prevista={names[pred[i]]} | conf={conf[i]:.2f}\n"
                     f"Grad-CAM>={a.cam_thr}: {st['bbox_w']}×{st['bbox_h']}px de {size}×{size}")
            panel(axs[r], img, cam, occ, title, patch)
            rows.append(dict(tipo=kind, image_id=ids[i], verdadeira=names[y[i]], prevista=names[pred[i]],
                             confianca=float(conf[i]), queda_max_oclusao=drop, **st, **image_attrs(paths[i]),
                             diagnostico=""))
        fig.tight_layout()
        fig.savefig(out / f"galeria_{kind}s.png", dpi=150)
        plt.close(fig)
    gc.close()
    pd.DataFrame(rows).to_csv(out / "galeria_info.csv", index=False)
    print(f"Salvo em {out}/: galeria_erros.png, galeria_acertos.png, galeria_info.csv, candidatos_erros.csv")
    print("Preencham a coluna 'diagnostico' em galeria_info.csv (texto é de vocês).")


if __name__ == "__main__":
    main()
