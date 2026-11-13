"""PA3 - Parte 0: o piso.

(a) CNN pequena do zero  e  (b) ResNet-18 ImageNet com fine-tuning,
ambas só com rotulos_{budget}.csv (default 5 por classe = 105 imagens).

Seleção de época: SOMENTE no val. O test_public é avaliado uma única vez,
no checkpoint escolhido pelo val.

Uso (Colab):
    !unzip -q pa3_dados.zip -d dados
    !python part0.py --data dados --out resultados --budget 5 --seed 0

Para a Parte 2 (piso nos 3 orçamentos / 3 seeds):
    for b, s in [(1,0),(5,0),(5,1),(5,2),(20,0)]: --budget b --seed s --which ft
Cada execução de `ft` acrescenta uma linha em resultados/curva_orcamento.csv.
"""
import argparse, copy, csv, json, os, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ----------------------------------------------------------------- dados
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def find_csv(root, key, exact=None):
    cands = [p for p in Path(root).rglob("*.csv")]
    if exact:
        hit = [p for p in cands if p.stem == exact]
        if hit:
            return hit[0]
    hit = [p for p in cands if key in p.stem.lower()]
    if not hit:
        raise FileNotFoundError(f"nenhum CSV com '{key}' em {root}")
    return sorted(hit, key=lambda p: len(p.stem))[0]


def pick_col(df, options):
    for o in options:
        if o in df.columns:
            return o
    raise KeyError(f"nenhuma coluna de {options} em {list(df.columns)}")


def load_classes(path):
    df = pd.read_csv(path)
    print("classes.csv colunas:", list(df.columns))
    name_col = next((c for c in df.columns if df[c].dtype == object), df.columns[-1])
    id_col = next((c for c in df.columns if c != name_col and pd.api.types.is_integer_dtype(df[c])), None)
    if id_col is not None:
        df = df.sort_values(id_col)
    names = [str(x) for x in df[name_col]]
    id2idx = {int(i): k for k, i in enumerate(df[id_col])} if id_col else {}
    return names, id2idx


def read_split(csv_path, class_names, id2idx, img_index):
    df = pd.read_csv(csv_path)
    ic = pick_col(df, ["image_id", "image", "id", "file", "filename", "path"])
    lc = pick_col(df, ["label", "class", "species", "category", "class_name"])
    name2idx = {n: i for i, n in enumerate(class_names)}
    paths, ys = [], []
    for img, lab in zip(df[ic], df[lc]):
        key = str(img)
        p = img_index.get(key) or img_index.get(Path(key).stem)
        if p is None:
            raise FileNotFoundError(f"imagem {key} (de {csv_path.name}) não encontrada")
        if str(lab) in name2idx:
            y = name2idx[str(lab)]
        else:
            y = id2idx.get(int(lab), int(lab))
        paths.append(p); ys.append(y)
    return paths, np.array(ys)


def index_images(root):
    idx = {}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            idx[p.name] = p
            idx[p.stem] = p
    return idx


class DS(Dataset):
    def __init__(self, imgs, ys, tf):
        self.imgs, self.ys, self.tf = imgs, ys, tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.tf(self.imgs[i]), int(self.ys[i])


def preload(paths):
    out = []
    for p in paths:
        with Image.open(p) as im:
            out.append(im.convert("RGB").copy())
    return out


def make_tf(size, train):
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.3, 0.3, 0.2),
            T.RandomGrayscale(0.2),  # noturnas em infravermelho
            T.ToTensor(), T.Normalize(MEAN, STD)])
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


# ---------------------------------------------------------------- modelos
class SmallCNN(nn.Module):
    def __init__(self, n_classes, widths=(32, 64, 128, 256, 256)):
        super().__init__()
        layers, c = [], 3
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c = w
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(c, n_classes)

    def forward(self, x):
        x = self.features(x).mean((2, 3))
        return self.fc(self.drop(x))


def build_resnet18(n_classes):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------- treino/aval
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot, loss_sum, preds, ys, probs = 0, 0.0, [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        lo = model(x)
        loss_sum += F.cross_entropy(lo, y, reduction="sum").item()
        probs.append(lo.softmax(1).cpu()); preds.append(lo.argmax(1).cpu()); ys.append(y.cpu())
        tot += len(y)
    preds, ys, probs = torch.cat(preds).numpy(), torch.cat(ys).numpy(), torch.cat(probs).numpy()
    return dict(loss=loss_sum / tot, acc=accuracy_score(ys, preds),
                f1=f1_score(ys, preds, average="macro", labels=range(probs.shape[1]), zero_division=0),
                preds=preds, ys=ys, probs=probs)


def train(model, tr_loader, tr_eval_loader, val_loader, epochs, lr, wd, device, name):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * len(tr_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    hist, best, best_state = [], (-1, 1e9), None
    for ep in range(1, epochs + 1):
        model.train()
        run, n = 0.0, 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            run += loss.item() * len(y); n += len(y)
        tr = evaluate(model, tr_eval_loader, device)   # treino sem aumentação, modo eval
        va = evaluate(model, val_loader, device)
        hist.append(dict(epoch=ep, train_loss_aug=run / n, train_loss=tr["loss"], train_acc=tr["acc"],
                         val_loss=va["loss"], val_acc=va["acc"], val_f1=va["f1"]))
        key = (va["f1"], -va["loss"])
        if key > (best[0], -best[1]):
            best, best_state, best_ep = (va["f1"], va["loss"]), copy.deepcopy(model.state_dict()), ep
        print(f"[{name}] ep {ep:3d} | train loss {tr['loss']:.3f} acc {tr['acc']:.2f} | "
              f"val loss {va['loss']:.3f} acc {va['acc']:.2f} f1 {va['f1']:.3f}")
    model.load_state_dict(best_state)
    return pd.DataFrame(hist), best_ep


def plot_curves(h, best_ep, title, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h.epoch, h.train_loss, label="treino"); ax[0].plot(h.epoch, h.val_loss, label="val")
    ax[0].axvline(best_ep, ls="--", c="gray", label=f"época escolhida ({best_ep})")
    ax[0].set(xlabel="época", ylabel="cross-entropy", title="Perda"); ax[0].legend()
    ax[1].plot(h.epoch, h.train_acc, label="treino"); ax[1].plot(h.epoch, h.val_acc, label="val")
    ax[1].plot(h.epoch, h.val_f1, label="val F1 macro", ls=":")
    ax[1].axvline(best_ep, ls="--", c="gray")
    ax[1].set(xlabel="época", ylabel="métrica", title="Acurácia / F1"); ax[1].legend()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(ys, preds, names, title, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = len(names)
    cm = confusion_matrix(ys, preds, labels=range(k))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(names, rotation=90); ax.set_yticklabels(names)
    for i in range(k):
        for j in range(k):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set(xlabel="previsto", ylabel="verdadeiro", title=title)
    fig.colorbar(im, fraction=0.046, label="fração da linha"); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    return cm


def memorization_epoch(h):
    """Primeira época com acurácia de treino >= 0.99 e época de mínimo da val loss."""
    a = h[h.train_acc >= 0.99]
    return (int(a.epoch.iloc[0]) if len(a) else None), int(h.epoch[h.val_loss.idxmin()])


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="pasta com o conteúdo de pa3_dados.zip")
    ap.add_argument("--out", default="resultados")
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--which", choices=["both", "scratch", "ft"], default="both")
    ap.add_argument("--epochs_ft", type=int, default=40)
    ap.add_argument("--epochs_scratch", type=int, default=80)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    seed_all(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print("device:", dev)

    idx = index_images(a.data)
    names, id2idx = load_classes(find_csv(a.data, "classes", "classes"))
    K = len(names); print(f"{K} classes")
    tr_csv = find_csv(a.data, f"rotulos_{a.budget}", f"rotulos_{a.budget}")
    va_csv = find_csv(a.data, "val", "val")
    te_csv = find_csv(a.data, "test_public", "test_public")
    print("CSVs:", tr_csv.name, va_csv.name, te_csv.name)

    sp = {}
    for k, c in [("tr", tr_csv), ("va", va_csv), ("te", te_csv)]:
        p, y = read_split(c, names, id2idx, idx)
        sp[k] = (preload(p), y)
        print(f"{k}: {len(y)} imagens, {len(set(y))} classes")

    mk = lambda k, train, shuffle: DataLoader(DS(*sp[k], make_tf(a.size, train)), batch_size=a.bs,
                                              shuffle=shuffle, num_workers=a.workers)
    tr_l, tr_eval_l = mk("tr", True, True), mk("tr", False, False)
    va_l, te_l = mk("va", False, False), mk("te", False, False)

    runs = []
    if a.which in ("both", "scratch"):
        runs.append(("scratch", SmallCNN(K), a.epochs_scratch, 2e-3, 5e-2))
    if a.which in ("both", "ft"):
        runs.append(("ft", build_resnet18(K), a.epochs_ft, 5e-4, 1e-2))

    for name, model, ep, lr, wd in runs:
        tag = f"{name}_b{a.budget}_s{a.seed}"
        print(f"\n=== {tag} | parâmetros: {n_params(model):,} ===")
        t0 = time.time()
        hist, best_ep = train(model, tr_l, tr_eval_l, va_l, ep, lr, wd, dev, name)
        hist.to_csv(out / f"hist_{tag}.csv", index=False)
        mem_ep, minvl_ep = memorization_epoch(hist)
        te = evaluate(model, te_l, dev)          # única avaliação no test_public
        va = evaluate(model, va_l, dev)
        title = {"scratch": "CNN do zero", "ft": "ResNet-18 ImageNet + FT"}[name]
        plot_curves(hist, best_ep, f"{title} ({a.budget}/classe, seed {a.seed})", out / f"curva_{tag}.png")
        cm = plot_confusion(te["ys"], te["preds"], names, f"{title}: test_public (F1 macro {te['f1']:.3f})",
                            out / f"confusao_{tag}.png")
        per_class = f1_score(te["ys"], te["preds"], average=None, labels=range(K), zero_division=0)
        pd.DataFrame(dict(classe=names, f1=per_class, n_test=cm.sum(1))).to_csv(
            out / f"f1_por_classe_{tag}.csv", index=False)
        res = dict(method=name, budget=a.budget, seed=a.seed, params=n_params(model),
                   best_epoch_val=best_ep, epoch_train_acc_99=mem_ep, epoch_min_val_loss=minvl_ep,
                   val_f1=va["f1"], val_acc=va["acc"], test_f1=te["f1"], test_acc=te["acc"],
                   train_minutes=(time.time() - t0) / 60)
        json.dump(res, open(out / f"metricas_{tag}.json", "w"), indent=2)
        print(json.dumps(res, indent=2))
        torch.save(model.state_dict(), out / f"pesos_{tag}.pt")
        if name == "ft":   # formato do curva_orcamento.csv (Parte 2)
            cp = out / "curva_orcamento.csv"
            new = not cp.exists()
            with open(cp, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["metodo", "orcamento", "seed", "f1_macro", "acuracia"])
                w.writerow(["piso", a.budget, a.seed, f"{te['f1']:.4f}", f"{te['acc']:.4f}"])


if __name__ == "__main__":
    main()
