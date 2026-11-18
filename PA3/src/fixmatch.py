"""Parte 1, Trilha B — FixMatch com os rótulos do orçamento + train inteiro como não rotulado.

Backbone/data herdados do part0_ft selecionado. Correções de desbalanceamento (fix.corr):
  none: limiar global tau | da: alinhamento de distribuição (ReMixMatch)
  flex: limiar por classe (FlexMatch, com piso fix.flex_floor) | both: da + flex

Diagnósticos no test_public (precisão/cobertura por classe vs tau, viés dos pseudo-rótulos) só são gerados
quando evaluate_test=True. A escolha de tau/correção é feita no val (src/sweep.py).

  python -m src.fixmatch --data dados --budget 5 --seed 0 --allow_defaults
"""
import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import transforms as T

from . import common as C
from . import preproc
from .models import build_model, count_params


# ------------------------------------------------------------------ dados
class WeakDS(Dataset):
    def __init__(self, arr, y, size, scale_min):
        self.arr, self.y = arr, y
        self.tf = T.Compose([T.RandomResizedCrop(size, scale=(scale_min, 1.0)), T.RandomHorizontalFlip(),
                             T.ToTensor(), T.Normalize(preproc.MEAN, preproc.STD)])

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.arr[i])), int(self.y[i])


class UnlDS(Dataset):
    """Mesmo recorte para as duas vistas: forte = fraca + RandAugment + apagamento. Com animais minúsculos,
    recortes independentes poderiam deixar o animal fora de uma das vistas e o pseudo-rótulo perderia o sentido."""

    def __init__(self, arr, size, aug):
        self.arr = arr
        self.geo = T.Compose([T.RandomResizedCrop(size, scale=(aug["weak_scale_min"], 1.0)),
                              T.RandomHorizontalFlip()])
        self.norm = T.Compose([T.ToTensor(), T.Normalize(preproc.MEAN, preproc.STD)])
        self.ra = T.RandAugment(num_ops=aug["randaug_ops"], magnitude=aug["randaug_mag"])
        self.erase = T.RandomErasing(p=aug["erase_p"], scale=tuple(aug["erase_scale"]), value="random")

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, i):
        base = self.geo(Image.fromarray(self.arr[i]))
        return self.norm(base), self.erase(self.norm(self.ra(base))), i


class EMA:
    def __init__(self, model, decay):
        self.m, self.d = copy.deepcopy(model).eval(), decay
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.m.parameters(), model.parameters()):
            e.mul_(self.d).add_(p.detach(), alpha=1 - self.d)
        for eb, b in zip(self.m.buffers(), model.buffers()):
            eb.copy_(b)


# --------------------------------------------------------------- FixMatch
def flex_thr(beta, tau, floor):
    """Limiar por classe do FlexMatch (convexo M(x)=x/(2-x)) com piso: sem ele o limiar é 0 no início."""
    return torch.clamp(tau * beta / (2 - beta), min=min(floor, tau))


def align_dist(p, pbar, m):
    """Alinhamento de distribuição: p <- normaliza(p / média_móvel(p)). Retorna (p_alinhado, pbar_novo)."""
    pbar = m * pbar + (1 - m) * p.mean(0)
    q = p / pbar
    return q / q.sum(1, keepdim=True), pbar


def train_loop(model, lab_l, unl_l, val_l, K, n_unl, cfg, dev, verbose=True):
    tr, fx = cfg["train"], cfg["fix"]
    steps, B = tr["steps"], min(tr["bs"], len(lab_l.dataset))
    use_amp = dev == "cuda"
    model.to(dev)
    ema = EMA(model, tr["ema"])
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.SGD([dict(params=decay, weight_decay=tr["wd"]), dict(params=no_decay, weight_decay=0)],
                          lr=tr["lr"], momentum=tr["momentum"], nesterov=True)
    # cosseno do FixMatch (Sohn et al., 2020): lr * cos(7*pi*k / (16*K))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, math.cos(7 * math.pi * s / (16 * steps))))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    do_da, do_flex = fx["corr"] in ("da", "both"), fx["corr"] in ("flex", "both")
    pbar = torch.full((K,), 1.0 / K, device=dev)
    status = torch.full((n_unl,), -1, dtype=torch.long, device=dev)   # FlexMatch: última classe com conf>=tau
    pl_used = torch.zeros(K, device=dev)

    def cur_beta():
        sigma = torch.bincount(status[status >= 0], minlength=K).float()
        return sigma / torch.clamp(torch.maximum(sigma.max(), n_unl - sigma.sum()), min=1.0)

    hist, best = [], dict(key=(-1.0, -1e9))
    lab_it, unl_it = iter(lab_l), iter(unl_l)
    run = dict(loss=0.0, lx=0.0, lu=0.0, mask=0.0, n=0)
    warm = max(1, min(5, steps // 2))
    t0 = t_steady = time.time()
    for step in range(1, steps + 1):
        model.train()
        xl, yl = next(lab_it)
        uw, us, ui = next(unl_it)
        xl, yl, uw, us, ui = (t.to(dev) for t in (xl, yl, uw, us, ui))
        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_amp):
            logits = model(torch.cat([xl, uw, us]))
        logits = logits.float()
        lw, ls = logits[B:].chunk(2)
        Lx = F.cross_entropy(logits[:B], yl)
        with torch.no_grad():
            p = lw.softmax(1)
            if do_da:
                p, pbar = align_dist(p, pbar, fx["da_m"])
            conf, pl = p.max(1)
            if do_flex:
                thr = flex_thr(cur_beta(), fx["tau"], fx["flex_floor"])
                mask = conf >= thr[pl]
                status[ui] = torch.where(conf >= fx["tau"], pl, torch.full_like(pl, -1))
            else:
                mask = conf >= fx["tau"]
            pl_used += torch.bincount(pl[mask], minlength=K).float()
        Lu = (F.cross_entropy(ls, pl, reduction="none") * mask.float()).mean()
        loss = Lx + fx["lam"] * Lu
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema.update(model)
        if step == warm:
            t_steady = time.time()
        run["loss"] += loss.item(); run["lx"] += Lx.item(); run["lu"] += Lu.item()
        run["mask"] += mask.float().mean().item(); run["n"] += 1
        if step % tr["eval_every"] == 0 or step == steps:
            va = C.evaluate(ema.m, val_l, dev)
            n = run["n"]
            row = dict(step=step, loss=run["loss"] / n, lx=run["lx"] / n, lu=run["lu"] / n,
                       mask_rate=run["mask"] / n, val_loss=va["loss"], val_acc=va["acc"], val_f1=va["f1"])
            hist.append(row)
            run = dict(loss=0.0, lx=0.0, lu=0.0, mask=0.0, n=0)
            if verbose:
                print(f"[fixmatch] step {step:5d} | loss {row['loss']:.3f} (Lx {row['lx']:.3f} Lu {row['lu']:.3f}) "
                      f"mask {row['mask_rate']:.2f} | val f1 {va['f1']:.3f} acc {va['acc']:.3f} | "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if (va["f1"], -va["loss"]) > best["key"]:
                best = dict(key=(va["f1"], -va["loss"]), step=step, state=copy.deepcopy(ema.m.state_dict()),
                            pbar=pbar.clone().cpu().numpy(), beta=cur_beta().cpu().numpy())
    sec_per_step = (time.time() - t_steady) / (steps - warm) if steps > warm else float("nan")
    ema.m.load_state_dict(best["state"])
    return pd.DataFrame(hist), best, pl_used.cpu().numpy(), ema, sec_per_step


# ----------------------------------------------------------- diagnósticos
def tau_diagnostic(probs, ys, K, taus, pbar, beta, fx, raw=False):
    """Roda o pseudo-rotulador sobre o test_public como se não tivesse rótulo (só diagnóstico).
    raw=True: limiar único tau sobre a confiança bruta (sem DA nem FlexMatch). Com flex, o limiar efetivo é
    max(tau*beta/(2-beta), flex_floor); se beta é pequeno ele vira o piso e a curva fica plana em tau (por isso o bruto)."""
    if raw:
        fx = dict(fx, corr="none")
    p = probs.copy()
    if fx["corr"] in ("da", "both"):
        p = p / pbar
        p = p / p.sum(1, keepdims=True)
    conf, pred = p.max(1), p.argmax(1)
    rows = []
    for tau in taus:
        if fx["corr"] in ("flex", "both"):
            m = conf >= np.maximum(tau * beta / (2 - beta), min(fx["flex_floor"], tau))[pred]
        else:
            m = conf >= tau
        for c in range(K):
            sel = m & (pred == c)
            n_pl, n_ok = int(sel.sum()), int((sel & (ys == c)).sum())
            rows.append(dict(tau=tau, classe_id=c, n_true=int((ys == c).sum()), n_pl=n_pl, n_correct=n_ok,
                             precision=n_ok / n_pl if n_pl else np.nan,
                             coverage=int((m & (ys == c)).sum()) / max(int((ys == c).sum()), 1)))
    return pd.DataFrame(rows)


def plot_diag(df, names, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    K, taus = len(names), sorted(df.tau.unique())
    order = df[df.tau == taus[0]].sort_values("n_true", ascending=False).classe_id.tolist()

    def mat(col):
        m = np.full((K, len(taus)), np.nan)
        for j, t in enumerate(taus):
            d = df[df.tau == t].set_index("classe_id")
            m[:, j] = [d.loc[c, col] for c in order]
        return m

    g = df.groupby("tau")[["n_pl", "n_correct"]].sum()
    n_tot = df[df.tau == taus[0]].n_true.sum()
    fig, ax = plt.subplots(1, 3, figsize=(17, 6), gridspec_kw=dict(width_ratios=[1, 1.2, 1.2]))
    ax[0].plot(taus, g.n_correct / g.n_pl.clip(lower=1), "o-", label="precisão (global)")
    ax[0].plot(taus, g.n_pl / n_tot, "s-", label="cobertura (global)")
    ax[0].set(xlabel=r"limiar $\tau$", ylim=(0, 1.02), title="Global")
    ax[0].legend()
    ax[0].grid(alpha=.3)
    lab = [f"{names[c]} (n={int(df[df.classe_id == c].n_true.iloc[0])})" for c in order]
    for k, (col, ttl) in enumerate([("precision", "Precisão do pseudo-rótulo"), ("coverage", "Cobertura por classe")]):
        im = ax[k + 1].imshow(mat(col), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax[k + 1].set_xticks(range(len(taus)))
        ax[k + 1].set_xticklabels(taus)
        ax[k + 1].set_yticks(range(K))
        ax[k + 1].set_yticklabels(lab, fontsize=7)
        ax[k + 1].set(xlabel=r"limiar $\tau$", title=ttl)
        fig.colorbar(im, ax=ax[k + 1], fraction=.046)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def share_report(dg, tau_ref, pl_used, names, path_png, path_csv, title, top_k):
    """Os pseudo-rótulos vão para as classes grandes? Fração real x fração de pseudo-rótulos."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = dg[np.isclose(dg.tau, tau_ref)].sort_values("n_true", ascending=False).reset_index(drop=True)
    true_sh = d.n_true / d.n_true.sum()
    pl_sh = d.n_pl / max(d.n_pl.sum(), 1)
    tr_sh = pl_used[d.classe_id.values] / max(pl_used.sum(), 1)
    out = pd.DataFrame(dict(classe=[names[c] for c in d.classe_id], n_true_test=d.n_true, share_real=true_sh,
                            share_pl_test=pl_sh, share_pl_treino=tr_sh, precisao_pl_test=d.precision))
    out.to_csv(path_csv, index=False)
    x, w = np.arange(len(d)), 0.28
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(x - w, true_sh, w, label="real (test_public)")
    ax.bar(x, pl_sh, w, label=f"pseudo-rótulos no test_public (tau={tau_ref})")
    ax.bar(x + w, tr_sh, w, label="pseudo-rótulos usados no treino")
    ax.set_xticks(x)
    ax.set_xticklabels(out.classe, rotation=75, ha="right", fontsize=8)
    ax.set(ylabel="fração", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    plt.close(fig)
    k = list(range(min(top_k, len(d))))
    return {f"top{top_k}_share_real": float(true_sh[k].sum()), f"top{top_k}_share_pl_test": float(pl_sh[k].sum()),
            f"top{top_k}_share_pl_treino": float(tr_sh[k].sum())}


# ------------------------------------------------------------------- run
def run(cfg, data, budget, seed, out_dir=None, evaluate_test=False, device=None, verbose=True, method="fixmatch"):
    assert data.unl is not None, "FixMatch precisa de load_data(need_unl=True)"
    assert evaluate_test is False or data.test is not None
    assert evaluate_test or data.test is None, "modo seleção: não carregue o test_public"
    dev = C.get_device(device)
    C.seed_all(seed)
    tr, fx, aug, size, K = cfg["train"], cfg["fix"], cfg["aug"], data.size, data.K
    m = cfg["model"]
    model = build_model(m["arch"], K, m.get("pretrained", False))
    B, mu, w, steps = min(tr["bs"], len(data.lab_y)), tr["mu"], cfg.get("workers", 0), tr["steps"]
    lab_ds = WeakDS(data.lab_train, data.lab_y, size, aug["weak_scale_min"])
    unl_ds = UnlDS(data.unl, size, aug)
    lab_l = DataLoader(lab_ds, batch_size=B, drop_last=True, num_workers=w, persistent_workers=w > 0,
                       sampler=RandomSampler(lab_ds, replacement=True, num_samples=steps * B))
    unl_l = DataLoader(unl_ds, batch_size=B * mu, drop_last=True, num_workers=w, persistent_workers=w > 0,
                       sampler=RandomSampler(unl_ds, replacement=True, num_samples=steps * B * mu))
    va_l = C.make_loader(data.val, data.val_y, C.EvalTF(), 64, False)
    timer = C.Timer()
    hist, best, pl_used, ema, sps = train_loop(model, lab_l, unl_l, va_l, K, len(data.unl), cfg, dev, verbose)
    va = C.evaluate(ema.m, va_l, dev)
    res = dict(method=method, budget=budget, seed=seed, cfg_hash=C.cfg_hash(cfg), cfg_source=cfg.get("_source"),
               params=count_params(model), best_step=best["step"], val_f1=va["f1"], val_acc=va["acc"],
               val_loss=va["loss"], mask_rate_last=float(hist.mask_rate.iloc[-1]), sec_per_step=sps,
               minutes=timer.minutes())
    if evaluate_test:
        te = C.evaluate(ema.m, C.make_loader(data.test, data.test_y, C.EvalTF(), 64, False), dev)  # 1 única vez
        res.update(test_f1=te["f1"], test_acc=te["acc"])
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        hist.to_csv(out / "hist.csv", index=False)
        C.save_json(cfg, out / "config.json")
        pd.DataFrame(dict(classe=data.names, pseudo_rotulos_usados=pl_used)).to_csv(out / "pl_usados_treino.csv",
                                                                                  index=False)
        if evaluate_test:
            title = f"FixMatch ({fx['corr']}, tau={fx['tau']}, {budget}/classe, seed {seed})"
            C.plot_confusion(te["ys"], te["preds"], data.names, f"{title}: test_public (F1 {te['f1']:.3f})",
                             out / "confusao.png")
            C.per_class_table(te["ys"], te["preds"], data.names).to_csv(out / "f1_por_classe.csv", index=False)
            np.save(out / "test_probs.npy", te["probs"])
            taus = sorted(set(fx["diag_taus"] + [fx["tau"]]))
            dg = tau_diagnostic(te["probs"], te["ys"], K, taus, best["pbar"], best["beta"], fx)
            dg.insert(2, "classe", [data.names[c] for c in dg.classe_id])
            dg.to_csv(out / "diag_tau.csv", index=False)
            plot_diag(dg, data.names, out / "diag_tau.png", f"Pseudo-rotulador no test_public ({fx['corr']})")
            dr = tau_diagnostic(te["probs"], te["ys"], K, taus, best["pbar"], best["beta"], fx, raw=True)
            dr.insert(2, "classe", [data.names[c] for c in dr.classe_id])
            dr.to_csv(out / "diag_tau_bruto.csv", index=False)
            plot_diag(dr, data.names, out / "diag_tau_bruto.png", "Pseudo-rotulador no test_public (confiança bruta >= tau)")
            res.update(share_report(dg, fx["tau"], pl_used, data.names, out / "vies_pseudorotulos.png",
                                    out / "vies_pseudorotulos.csv", f"Fração por classe ({fx['corr']})",
                                    fx["share_top_k"]))
        C.save_run_checkpoint(out / "weights.pt", ema.m, cfg, data, meta=dict(res))
        C.save_json(res, out / "metrics.json")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="resultados")
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow_defaults", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="a.b=v")
    a = ap.parse_args()
    cfg = C.method_cfg("fixmatch", a.budget, a.set, a.allow_defaults)
    print("config:", cfg["_source"])
    data = C.DataCache(a.data, a.cache).get(a.budget, cfg, need_unl=True, need_test=True)
    res = run(cfg, data, a.budget, a.seed, C.run_dir(a.out, "fixmatch", a.budget, a.seed), True)
    C.append_curve_row(Path(a.out) / "curva_orcamento.csv", C.project()["rotulo_no_csv"]["fixmatch"], a.budget,
                       a.seed, res["test_f1"], res["test_acc"])
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})


if __name__ == "__main__":
    main()
