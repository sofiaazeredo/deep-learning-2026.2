"""Parte 3 — mapas de saliência: Grad-CAM e oclusão (sem dependências extras).

  python -m src.saliencia --ckpt pesos/modelo_final.pt --imagem foto.jpg --out mapa.png
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from . import preproc
from .models import load_checkpoint, target_layer


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.act, self.grad = model, None, None
        self.h = layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        if out.requires_grad:               # forwards sob no_grad (ex.: oclusão) não participam do Grad-CAM
            self.act = out
            out.register_hook(lambda g: setattr(self, "grad", g))

    def close(self):
        self.h.remove()

    def __call__(self, x, target=None):
        """x: (1,3,H,W) já normalizado. Retorna (cam HxW em [0,1], probs (K,), classe usada)."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)       # garante grafo mesmo se os parâmetros estiverem congelados
        logits = self.model(x)
        t = int(logits.argmax(1)) if target is None else int(target)
        logits[0, t].backward()
        w = self.grad.mean((2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), logits.softmax(1)[0].detach().cpu().numpy(), t


@torch.no_grad()
def occlusion_map(model, x, target, patch, stride, bs=32):
    """Queda da probabilidade da classe `target` ao cobrir cada janela patch×patch com a média do ImageNet
    (zero após normalizar). Retorna (mapa HxW em [0,1], maior queda absoluta)."""
    model.eval()
    _, _, H, W = x.shape
    base = model(x).softmax(1)[0, target].item()
    ys = list(range(0, max(H - patch, 0) + 1, stride))
    xs = list(range(0, max(W - patch, 0) + 1, stride))
    heat, cnt = torch.zeros(H, W), torch.zeros(H, W)
    boxes = [(i, j) for i in ys for j in xs]
    for k in range(0, len(boxes), bs):
        chunk = boxes[k:k + bs]
        xb = x.repeat(len(chunk), 1, 1, 1)
        for n, (i, j) in enumerate(chunk):
            xb[n, :, i:i + patch, j:j + patch] = 0.0
        drops = base - model(xb).softmax(1)[:, target]
        for n, (i, j) in enumerate(chunk):
            heat[i:i + patch, j:j + patch] += drops[n].item()
            cnt[i:i + patch, j:j + patch] += 1
    heat = (heat / cnt.clamp(min=1)).clamp(min=0)
    mx = float(heat.max())
    return (heat / (mx + 1e-8)).numpy(), mx


def cam_stats(cam, thr):
    """Resumo numérico do mapa (coordenadas do tensor de entrada, ex.: 224×224). thr é só de visualização."""
    m = cam >= thr
    ys, xs = np.where(m)
    H, W = cam.shape
    py, px = np.unravel_index(int(cam.argmax()), cam.shape)
    return dict(area_frac=float(m.mean()), bbox_w=int(xs.max() - xs.min() + 1), bbox_h=int(ys.max() - ys.min() + 1),
                cx=float(xs.mean() / W), cy=float(ys.mean() / H), peak_x=float(px / W), peak_y=float(py / H))


def overlay(img_uint8, cam, alpha=0.45):
    import matplotlib
    matplotlib.use("Agg")
    heat = (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.float32)
    return np.clip((1 - alpha) * img_uint8.astype(np.float32) + alpha * heat, 0, 255).astype(np.uint8)


def input_view(path, size, mode):
    """A imagem como o modelo a vê (após resize/pad), em uint8, e o tensor normalizado correspondente."""
    im = preproc.resize_square(preproc.load_rgb(path), size, mode)
    return np.asarray(im), preproc.to_tensor(im)[None]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--imagem", required=True)
    ap.add_argument("--out", default="saliencia.png")
    ap.add_argument("--target", type=int, default=None, help="índice da classe (default: a prevista)")
    ap.add_argument("--cam_thr", type=float, default=0.5, help="limiar só para o bbox/área reportados")
    ap.add_argument("--patch", type=int, default=None, help="janela da oclusão (default: size//8)")
    ap.add_argument("--stride", type=int, default=None, help="passo da oclusão (default: patch//2)")
    a = ap.parse_args()
    model, ck = load_checkpoint(a.ckpt)
    size, mode = ck["preprocess"]["size"], ck["preprocess"]["resize_mode"]
    img, x = input_view(a.imagem, size, mode)
    gc = GradCAM(model, target_layer(model, ck["arch"]))
    cam, probs, t = gc(x, a.target)
    gc.close()
    patch = a.patch or size // 8
    occ, _ = occlusion_map(model, x, t, patch, a.stride or patch // 2)
    st = cam_stats(cam, a.cam_thr)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for axx, im, ttl in zip(ax, [img, overlay(img, cam), overlay(img, occ)],
                            ["entrada do modelo", "Grad-CAM", f"oclusão (janela {patch}px)"]):
        axx.imshow(im)
        axx.set_title(ttl)
        axx.axis("off")
    fig.suptitle(f"{ck['class_names'][t]}  p={probs[t]:.2f}  | região Grad-CAM>={a.cam_thr}: "
                 f"{st['bbox_w']}×{st['bbox_h']} px de {size}×{size}")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(st)


if __name__ == "__main__":
    main()
