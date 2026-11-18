"""Utilitários compartilhados: config, dados, treino supervisionado, métricas e gráficos."""
import copy
import hashlib
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from . import preproc
from .models import build_model, count_params, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / "configs" / "selected"
DEFAULTS = ROOT / "configs" / "defaults"


# =============================================================== config
def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _jd(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def save_json(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_jd)


def project():
    return load_json(ROOT / "configs" / "project.json")


def deep_update(base, new):
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def cfg_set(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def cfg_get(cfg, dotted, default=None):
    d = cfg
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        out[k] = v
    return out


def apply_overrides(cfg, items):
    cfg = copy.deepcopy(cfg)
    ov = items if isinstance(items, dict) else parse_overrides(items)
    for k, v in ov.items():
        cfg_set(cfg, k, v)
    return cfg


def cfg_hash(cfg):
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.md5(json.dumps(clean, sort_keys=True).encode()).hexdigest()[:10]


def _backbone_from_part0(allow_defaults):
    p = SELECTED / "part0_ft_b5.json"
    if p.exists():
        c = load_json(p)
        return {"model": c["model"], "data": c["data"]}, str(p)
    if allow_defaults:
        c = load_json(DEFAULTS / "part0_ft.json")
        return {"model": c["model"], "data": c["data"]}, "DEFAULTS (não validados)"
    raise SystemExit("Falta configs/selected/part0_ft_b5.json (o backbone é o do piso). Rode primeiro:\n"
                     "  python -m src.sweep --method part0_ft --budget 5 --data dados\n"
                     "ou passe --allow_defaults para usar os defaults NÃO validados.")


def method_cfg(method, budget, overrides=(), allow_defaults=False, use_selected=True):
    """Config de um método: selected (escolhida no val) > defaults (só com --allow_defaults ou em sweeps)."""
    sel = SELECTED / f"{method}_b{budget}.json"
    from_defaults = False
    if use_selected and sel.exists():
        cfg, src = load_json(sel), str(sel)
    elif use_selected and method == "fixmatch" and (SELECTED / "fixmatch_b5.json").exists():
        cfg, src = load_json(SELECTED / "fixmatch_b5.json"), f"{SELECTED / 'fixmatch_b5.json'} (reaproveitado do orçamento 5)"
    elif allow_defaults or not use_selected:
        cfg, src, from_defaults = load_json(DEFAULTS / f"{method}.json"), "DEFAULTS (não validados)", True
    else:
        raise SystemExit(f"Falta {sel}. Rode o sweep ({method}, orçamento {budget}) ou use --allow_defaults "
                         "(valores NÃO validados).")
    if method == "fixmatch" and from_defaults:
        bb, bsrc = _backbone_from_part0(allow_defaults)
        cfg = deep_update(cfg, bb)
        src += f" + backbone de {bsrc}"
    cfg = apply_overrides(cfg, overrides)
    cfg["_source"] = src
    return cfg


# ================================================================ dados
def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def get_device(name=None):
    return name or ("cuda" if torch.cuda.is_available() else "cpu")


def load_schema():
    p = ROOT / "configs" / "data_schema.json"
    if not p.exists():
        raise SystemExit("configs/data_schema.json não existe. Rode primeiro:\n"
                         "  python -m tests.check_data --data dados")
    return load_json(p)


def find_file(data_dir, fname):
    hits = list(Path(data_dir).rglob(fname))
    if not hits:
        raise FileNotFoundError(f"{fname} não encontrado em {data_dir}")
    return min(hits, key=lambda p: len(p.parts))


def index_images(data_dir):
    idx = {}
    for p in Path(data_dir).rglob("*"):
        if p.suffix.lower() in preproc.IMG_EXT:
            idx[p.name] = p
            idx.setdefault(p.stem, p)
    return idx


def load_classes(data_dir, schema):
    df = pd.read_csv(find_file(data_dir, schema["files"]["classes"]))
    cc = schema["classes_cols"]
    if cc.get("id"):
        df = df.sort_values(cc["id"]).reset_index(drop=True)
    names = [str(x) for x in df[cc["name"]]]
    id2idx = {int(v): i for i, v in enumerate(df[cc["id"]])} if cc.get("id") else {}
    return names, id2idx


def read_split(data_dir, schema, split, names, id2idx, budget=None, img_index=None):
    """split in {train, rotulos, val, test_public}. Retorna (ids, paths, y|None)."""
    f = schema["files"]
    if split == "train" and not f.get("train"):         # pool = pasta (pacote real: images/train/)
        fs = sorted(p for p in (Path(data_dir) / schema["train_dir"]).iterdir() if p.suffix.lower() in preproc.IMG_EXT)
        return [p.name if schema.get("id_has_extension") else p.stem for p in fs], fs, None
    fname = f["rotulos"].format(b=budget) if split == "rotulos" else f[split]
    df = pd.read_csv(find_file(data_dir, fname))
    img_index = img_index or index_images(data_dir)
    ic = schema["split_cols"]["image"]
    ids = [str(v) for v in df[ic]]
    paths = []
    for i in ids:
        p = img_index.get(i) or img_index.get(Path(i).stem)
        if p is None:
            raise FileNotFoundError(f"imagem '{i}' ({fname}) não encontrada em {data_dir}")
        paths.append(p)
    if split == "train":
        return ids, paths, None
    lc = schema["split_cols"]["label"]
    if schema["label_kind"] == "name":
        n2i = {n: i for i, n in enumerate(names)}
        y = np.array([n2i[str(v)] for v in df[lc]])
    else:
        y = np.array([id2idx[int(v)] if id2idx else int(v) for v in df[lc]])
    return ids, paths, y


def _load_resized(path, side, mode):
    return np.asarray(preproc.resize_square(preproc.load_rgb(path), side, mode))


def build_array(paths, side, mode, cache_dir=None, workers=8):
    """(N, side, side, 3) uint8. Um array só (em vez de lista de PIL) evita cópias nos workers do DataLoader."""
    cache = None
    if cache_dir:
        key = hashlib.md5(("|".join(map(str, paths)) + f"|{side}|{mode}").encode()).hexdigest()[:16]
        cache = Path(cache_dir) / f"{key}.npy"
        if cache.exists():
            return np.load(cache)
    with ThreadPoolExecutor(workers) as ex:
        arr = np.stack(list(ex.map(lambda p: _load_resized(p, side, mode), paths)))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, arr)
    return arr


class Data:
    """Tudo que um run precisa, já em memória. test só é carregado quando need_test=True."""
    pass


def load_data(data_dir, budget, size, mode, preload_factor=1.15, need_unl=False, need_test=False, cache_dir=None):
    schema = load_schema()
    names, id2idx = load_classes(data_dir, schema)
    idx = index_images(data_dir)
    d = Data()
    d.names, d.K, d.size, d.mode, d.budget = names, len(names), size, mode, budget
    d.s = int(math.ceil(size * preload_factor))
    rd = lambda split, **kw: read_split(data_dir, schema, split, names, id2idx, img_index=idx, **kw)
    d.lab_ids, d.lab_paths, d.lab_y = rd("rotulos", budget=budget)
    d.lab_train = build_array(d.lab_paths, d.s, mode, cache_dir)      # fonte das aumentações
    d.lab_eval = build_array(d.lab_paths, size, mode, cache_dir)      # treino sem aumentação (métricas)
    d.val_ids, d.val_paths, d.val_y = rd("val")
    d.val = build_array(d.val_paths, size, mode, cache_dir)
    d.test = d.test_ids = d.test_paths = d.test_y = None
    if need_test:
        d.test_ids, d.test_paths, d.test_y = rd("test_public")
        d.test = build_array(d.test_paths, size, mode, cache_dir)
    d.unl = d.unl_ids = None
    if need_unl:
        d.unl_ids, unl_paths, _ = rd("train")
        d.unl = build_array(unl_paths, d.s, mode, cache_dir)
    return d


class DataCache:
    """Guarda só o Data mais recente (sweeps que variam data.size/resize_mode trocam de entrada)."""

    def __init__(self, data_dir, cache_dir=None):
        self.data_dir, self.cache_dir, self.key, self.val = data_dir, cache_dir, None, None

    def get(self, budget, cfg, need_unl=False, need_test=False):
        dc = cfg["data"]
        key = (budget, dc["size"], dc["resize_mode"], dc.get("preload_factor", 1.15), need_unl, need_test)
        if key != self.key:
            self.val = None
            self.val = load_data(self.data_dir, budget, dc["size"], dc["resize_mode"], key[3], need_unl, need_test,
                                 self.cache_dir)
            self.key = key
        return self.val


# ============================================================ transforms
class EvalTF:
    def __call__(self, im):
        return preproc.to_tensor(im)


class ArrDS(Dataset):
    def __init__(self, arr, y, tf):
        self.arr, self.y, self.tf = arr, y, tf

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.arr[i])), int(self.y[i])


def train_tf(size, aug):
    ts = [T.RandomResizedCrop(size, scale=(aug["scale_min"], 1.0)), T.RandomHorizontalFlip(aug.get("hflip", 0.5))]
    if aug.get("jitter"):
        ts.append(T.ColorJitter(*aug["jitter"]))
    if aug.get("gray_p", 0) > 0:
        ts.append(T.RandomGrayscale(aug["gray_p"]))      # câmeras noturnas em infravermelho
    ts += [T.ToTensor(), T.Normalize(preproc.MEAN, preproc.STD)]
    return T.Compose(ts)


def make_loader(arr, y, tf, bs, shuffle, workers=0, drop_last=False):
    return DataLoader(ArrDS(arr, y, tf), batch_size=bs, shuffle=shuffle, drop_last=drop_last, num_workers=workers,
                      persistent_workers=workers > 0)


# ======================================================== treino/avaliação
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n, ls, preds, ys, probs = 0, 0.0, [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        lo = model(x).float()
        ls += F.cross_entropy(lo, y, reduction="sum").item()
        probs.append(lo.softmax(1).cpu())
        preds.append(lo.argmax(1).cpu())
        ys.append(y.cpu())
        n += len(y)
    preds, ys, probs = torch.cat(preds).numpy(), torch.cat(ys).numpy(), torch.cat(probs).numpy()
    return dict(loss=ls / n, acc=accuracy_score(ys, preds),
                f1=f1_score(ys, preds, average="macro", labels=range(probs.shape[1]), zero_division=0),
                preds=preds, ys=ys, probs=probs)


def fit_supervised(model, data, cfg, device, name="sup", verbose=True):
    """Treino supervisionado nos rótulos do orçamento. Época escolhida SÓ pelo val (F1 macro, desempate: loss)."""
    tr, size = cfg["train"], data.size
    w = cfg.get("workers", 0)
    n = len(data.lab_y)
    bs = min(tr["bs"], n)
    tr_l = make_loader(data.lab_train, data.lab_y, train_tf(size, cfg["aug"]), bs, True, w, drop_last=(n % bs == 1))
    tr_eval = make_loader(data.lab_eval, data.lab_y, EvalTF(), 64, False)
    va_l = make_loader(data.val, data.val_y, EvalTF(), 64, False)
    model.to(device)
    if tr["optimizer"] == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["wd"])
    elif tr["optimizer"] == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=tr["lr"], momentum=0.9, nesterov=True, weight_decay=tr["wd"])
    else:
        raise ValueError(tr["optimizer"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tr["lr"], total_steps=tr["epochs"] * len(tr_l),
                                                pct_start=tr["pct_start"])
    hist, best_key, best_state, best_ep = [], (-1.0, -1e9), None, 0
    for ep in range(1, tr["epochs"] + 1):
        model.train()
        run, cnt = 0.0, 0
        for x, y in tr_l:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item() * len(y)
            cnt += len(y)
        t, v = evaluate(model, tr_eval, device), evaluate(model, va_l, device)
        hist.append(dict(epoch=ep, train_loss_aug=run / cnt, train_loss=t["loss"], train_acc=t["acc"],
                         val_loss=v["loss"], val_acc=v["acc"], val_f1=v["f1"]))
        if (v["f1"], -v["loss"]) > best_key:
            best_key, best_ep, best_state = (v["f1"], -v["loss"]), ep, copy.deepcopy(model.state_dict())
        if verbose:
            print(f"[{name}] ep {ep:3d} | train loss {t['loss']:.3f} acc {t['acc']:.2f} | "
                  f"val loss {v['loss']:.3f} acc {v['acc']:.2f} f1 {v['f1']:.3f}", flush=True)
    model.load_state_dict(best_state)
    return pd.DataFrame(hist), best_ep


def memorization_epochs(h):
    """(1ª época com 100% de acerto no treino, época do mínimo da val loss). Sem limiar arbitrário."""
    full = h[h.train_acc >= 1.0]
    return (int(full.epoch.iloc[0]) if len(full) else None), int(h.epoch[h.val_loss.idxmin()])


# ============================================================ saídas
def run_dir(out, method, budget, seed):
    return Path(out) / "runs" / f"{method}_b{budget}_s{seed}"


def per_class_table(ys, preds, names):
    K = len(names)
    f1 = f1_score(ys, preds, average=None, labels=range(K), zero_division=0)
    return pd.DataFrame(dict(classe=names, f1=f1, n_test=np.bincount(ys, minlength=K)))


def preprocess_meta(data):
    return dict(size=data.size, resize_mode=data.mode)


def save_run_checkpoint(path, model, cfg, data, meta, project_cfg=None):
    lim = (project_cfg or project())["limite_parametros"]
    n = count_params(model)
    assert n <= lim, f"modelo com {n:,} parâmetros > limite {lim:,}"
    save_checkpoint(path, model, cfg["model"]["arch"], data.names, preprocess_meta(data), meta,
                    arch_kwargs={k: cfg["model"][k] for k in ("widths", "dropout") if k in cfg["model"]})


def append_curve_row(csv_path, metodo, budget, seed, f1, acc):
    """Idempotente: linha com a mesma chave (metodo, orcamento, seed) é substituída."""
    cols = ["metodo", "orcamento", "seed", "f1_macro", "acuracia"]
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame(columns=cols)
    df = df[~((df.metodo == metodo) & (df.orcamento == budget) & (df.seed == seed))]
    df = pd.concat([df, pd.DataFrame([[metodo, budget, seed, round(float(f1), 4), round(float(acc), 4)]], columns=cols)])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["metodo", "orcamento", "seed"]).to_csv(csv_path, index=False)


# ============================================================ gráficos
def plot_curves(h, best_ep, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h.epoch, h.train_loss, label="treino (sem aug)")
    ax[0].plot(h.epoch, h.val_loss, label="val")
    ax[0].axvline(best_ep, ls="--", c="gray", label=f"época escolhida ({best_ep})")
    ax[0].set(xlabel="época", ylabel="cross-entropy", title="Perda")
    ax[0].legend()
    ax[1].plot(h.epoch, h.train_acc, label="treino (sem aug)")
    ax[1].plot(h.epoch, h.val_acc, label="val")
    ax[1].plot(h.epoch, h.val_f1, ":", label="val F1 macro")
    ax[1].axvline(best_ep, ls="--", c="gray")
    ax[1].set(xlabel="época", ylabel="métrica", title="Acurácia / F1")
    ax[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(ys, preds, names, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = len(names)
    cm = confusion_matrix(ys, preds, labels=range(k))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticklabels(names)
    for i in range(k):
        for j in range(k):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set(xlabel="previsto", ylabel="verdadeiro", title=title)
    fig.colorbar(im, fraction=0.046, label="fração da linha")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def minutes(self):
        return (time.time() - self.t0) / 60
