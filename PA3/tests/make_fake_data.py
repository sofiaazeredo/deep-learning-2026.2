"""Gera um pacote SINTÉTICO com a mesma estrutura do pa3_dados.zip, para testar o pipeline sem os dados reais
(e sem GPU). NÃO tem relação com o ENA24: os números obtidos nele não significam nada.

  python -m tests.make_fake_data --out /content/fake --per_class_max 60
Se os CSVs reais tiverem outros nomes de colunas, os checks de schema são exercitados com --id_col/--label_col/--label_kind.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def make(root, per_class_max=60, seed=0, id_col="image_id", label_col="label", label_kind="name", with_ext=False,
         n_classes=21, val_per=3, test_per=5, real_layout=False):
    """real_layout=True imita o pacote real: images/{train,val,test_public}/, labels/*.csv, SEM train.csv."""
    rng = np.random.RandomState(seed)
    root = Path(root)
    csv_dir = root / "labels" if real_layout else root
    csv_dir.mkdir(parents=True, exist_ok=True)
    for sub in (["images/train", "images/val", "images/test_public"] if real_layout else ["imgs"]):
        (root / sub).mkdir(parents=True, exist_ok=True)
    names = ["bear", "horse", "bird", "american_crow"] + [f"especie_{i:02d}" for i in range(4, n_classes)]
    # desbalanceado: classe 0 grande, classe 1 pequena, o resto decresce (mín. 25 para caber o orçamento 20)
    sizes = [per_class_max, 25] + [int(max(25, per_class_max * (0.9 - 0.03 * i))) for i in range(2, n_classes)]
    pd.DataFrame({"class_id": range(n_classes), "class_name": names}).to_csv(root / "classes.csv", index=False)
    cams = [rng.randint(0, 255, 3) for _ in range(6)]

    def draw(c, name, sub="imgs"):
        w, h = 512, int(rng.choice([288, 341, 384, 512]))
        cam = cams[rng.randint(len(cams))]
        bg = (rng.rand(h // 16, w // 16, 3) * 60 + cam * 0.4).astype(np.uint8)
        im = Image.fromarray(bg).resize((w, h), Image.BILINEAR)
        d = ImageDraw.Draw(im)
        frac = rng.choice([0.02, 0.05, 0.1, 0.25])
        ew = max(2, int(w * np.sqrt(frac) * 0.5 * (0.7 + 0.3 * (c % 3))))
        eh = max(2, int(h * np.sqrt(frac) * 0.5 * (0.7 + 0.3 * ((c + 1) % 3))))
        cx, cy = rng.randint(ew + 1, w - ew - 1), rng.randint(eh + 1, h - eh - 1)
        col = tuple(int(v) for v in (np.array([(c * 47) % 256, (c * 91 + 60) % 256, (c * 23 + 120) % 256])))
        d.ellipse([cx - ew, cy - eh, cx + ew, cy + eh], fill=col)
        if rng.rand() < 0.3:
            im = im.convert("L").convert("RGB")
        im.save(root / ("images/" + sub if real_layout else "imgs") / f"{name}.jpg", quality=90)

    rows = {"train": [], "val": [], "test_public": []}
    pools = []
    for c in range(n_classes):
        ids = []
        for j in range(sizes[c]):
            iid = f"tr_{c:02d}_{j:03d}"
            draw(c, iid, "train")
            ids.append(iid)
        pools.append(ids)
        rows["train"] += [(i, c) for i in ids]
        for j in range(val_per if c != 1 else 2):
            iid = f"va_{c:02d}_{j:03d}"
            draw(c, iid, "val")
            rows["val"].append((iid, c))
        for j in range(test_per if c != 1 else 4):
            iid = f"te_{c:02d}_{j:03d}"
            draw(c, iid, "test_public")
            rows["test_public"].append((iid, c))

    def lab(c):
        return names[c] if label_kind == "name" else c

    def fid(i):
        return i + ".jpg" if with_ext else i

    for split in ("val", "test_public"):
        pd.DataFrame([(fid(i), lab(c)) for i, c in rows[split]], columns=[id_col, label_col]).to_csv(
            csv_dir / f"{split}.csv", index=False)
    if not real_layout:
        pd.DataFrame({id_col: [fid(i) for i, _ in rows["train"]]}).to_csv(root / "train.csv", index=False)
    # orçamentos aninhados: rotulos_1 ⊂ rotulos_5 ⊂ rotulos_20 ⊂ train
    order = [list(rng.permutation(p)) for p in pools]
    for b in (1, 5, 20):
        sel = [(order[c][k], c) for c in range(n_classes) for k in range(b)]
        pd.DataFrame([(fid(i), lab(c)) for i, c in sel], columns=[id_col, label_col]).to_csv(
            csv_dir / f"rotulos_{b}.csv", index=False)
    return root


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_class_max", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--id_col", default="image_id")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--label_kind", choices=["name", "id"], default="name")
    ap.add_argument("--with_ext", action="store_true")
    ap.add_argument("--real_layout", action="store_true", help="images/{train,val,test_public}/ + labels/, sem train.csv")
    a = ap.parse_args()
    make(a.out, a.per_class_max, a.seed, a.id_col, a.label_col, a.label_kind, a.with_ext, real_layout=a.real_layout)
    print("pacote sintético em", a.out)
