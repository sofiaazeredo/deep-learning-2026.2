"""Verifica o pacote de dados REAL contra o enunciado e DETECTA o schema (colunas), gravando
configs/data_schema.json. Nada de colunas/nomes adivinhados no código: tudo que os scripts assumem sai daqui.

  python -m tests.check_data --data dados            # real: confere contagens do enunciado
  python -m tests.check_data --data /content/fake --fake   # sintético: pula contagens exatas
Revise a saída (cabeçalhos dos CSVs e o schema detectado) antes de seguir.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import common as C  # noqa: E402
from src.preproc import IMG_EXT  # noqa: E402
from tests._report import Report  # noqa: E402

IMG_CANDS = ["image_id", "image", "id", "file", "filename", "file_name", "path", "image_path"]
LAB_CANDS = ["label", "class", "species", "category", "class_name", "label_name", "category_name", "y"]


def pick(df, cands, what, R, csv, forced=None):
    if forced:
        if R.check(forced in df.columns, f"coluna de {what} (forçada: {forced})", f"colunas: {list(df.columns)}"):
            return forced
        return None
    hit = [c for c in cands if c in df.columns]
    if len(hit) == 1:
        return hit[0]
    R.fail(f"coluna de {what} em {csv}", f"candidatas {hit or 'nenhuma'}; colunas: {list(df.columns)}. "
                                          f"Ajuste a lista de candidatas ou renomeie o CSV.")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fake", action="store_true", help="pacote sintético: não exige as contagens do enunciado")
    ap.add_argument("--no_write", action="store_true")
    ap.add_argument("--image_col", default=None, help="força o nome da coluna de imagem dos CSVs de split")
    ap.add_argument("--label_col", default=None, help="força o nome da coluna de rótulo dos CSVs de split")
    ap.add_argument("--max_size_scan", type=int, default=2000, help="nº de imagens para estatísticas de tamanho")
    a = ap.parse_args()
    P = C.project()
    R = Report("check_data")
    data = Path(a.data)

    files = dict(classes="classes.csv", train=None, rotulos="rotulos_{b}.csv", val="val.csv",
                 test_public="test_public.csv")
    paths = {}
    for k, f in files.items():
        if f is None:                      # o pool (train) é tratado abaixo: CSV opcional ou pasta
            continue
        for name in ([f.format(b=b) for b in P["orcamentos"]] if k == "rotulos" else [f]):
            hits = list(data.rglob(name))
            if R.check(bool(hits), f"arquivo {name}", str(min(hits, key=lambda p: len(p.parts))) if hits else "ausente"):
                paths[name] = min(hits, key=lambda p: len(p.parts))
    # pool não rotulado: train.csv se existir; senão a PASTA 'train' (é o caso do pacote real: images/train/)
    train_dir = None
    hits = list(data.rglob("train.csv"))
    if hits:
        files["train"] = "train.csv"
        paths["train.csv"] = min(hits, key=lambda p: len(p.parts))
        R.ok("pool não rotulado: train.csv", str(paths["train.csv"]))
    else:
        cands = [d for d in data.rglob("*") if d.is_dir() and d.name == "train"
                 and any(p.suffix.lower() in IMG_EXT for p in d.iterdir())]
        if R.check(len(cands) == 1, "pool não rotulado: pasta 'train' com imagens (não há train.csv)",
                   str(cands[0]) if len(cands) == 1 else f"encontradas {len(cands)}: {[str(c) for c in cands]}"):
            train_dir = cands[0]
    if R.rows and any(r[0] == "FAIL" for r in R.rows):
        R.finish()

    print("\n--- cabeçalhos (confira se o schema detectado faz sentido) ---")
    dfs = {n: pd.read_csv(p) for n, p in paths.items()}
    for n, d in dfs.items():
        print(f"{n}: {len(d)} linhas | colunas {list(d.columns)}")
        print(d.head(2).to_string(index=False))

    # ---- classes.csv
    cdf = dfs["classes.csv"]
    str_cols = [c for c in cdf.columns if pd.api.types.is_string_dtype(cdf[c]) and not pd.api.types.is_numeric_dtype(cdf[c])]
    int_cols = [c for c in cdf.columns if pd.api.types.is_integer_dtype(cdf[c])]
    name_col = str_cols[0] if len(str_cols) == 1 else None
    id_col_cls = int_cols[0] if len(int_cols) == 1 else None
    R.check(name_col is not None, "coluna de nome em classes.csv", f"{name_col}" if name_col else f"colunas {list(cdf.columns)}")
    if name_col is None:
        R.finish()
    names_sorted = [str(x) for x in (cdf.sort_values(id_col_cls)[name_col] if id_col_cls else cdf[name_col])]
    R.check(len(names_sorted) == P["n_classes"], "nº de classes", f"{len(names_sorted)} (enunciado: {P['n_classes']})")
    R.check(len(set(names_sorted)) == len(names_sorted), "nomes de classe únicos")
    R.info(f"id da classe: {id_col_cls or 'ausente (ordem das linhas)'} | classes: {names_sorted}")

    # ---- colunas dos splits
    ref = dfs["val.csv"]
    img_col = pick(ref, IMG_CANDS, "imagem", R, "val.csv", a.image_col)
    lab_col = pick(ref, LAB_CANDS, "rótulo", R, "val.csv", a.label_col)
    if img_col is None or lab_col is None:
        R.finish()
    for n, d in dfs.items():
        if n == "classes.csv":
            continue
        R.check(img_col in d.columns, f"{n}: tem '{img_col}'")
        if n != "train.csv":
            R.check(lab_col in d.columns, f"{n}: tem '{lab_col}'")
    if "train.csv" in dfs and lab_col in dfs["train.csv"].columns:
        R.warn("train.csv tem coluna de rótulo", "NÃO será usada (os únicos rótulos do train são os dos rotulos_*.csv)")
    label_kind = "id" if pd.api.types.is_integer_dtype(ref[lab_col]) else "name"
    valid = set(names_sorted) if label_kind == "name" else set(int(v) for v in (cdf[id_col_cls] if id_col_cls else range(len(cdf))))
    for n in [k for k in dfs if k not in ("classes.csv", "train.csv")]:
        bad = set(dfs[n][lab_col]) - valid
        R.check(not bad, f"{n}: rótulos válidos ({label_kind})", f"inválidos: {sorted(bad)[:5]}" if bad else "")

    # ---- imagens
    idx = C.index_images(data)
    sample_id = str(ref[img_col].iloc[0])
    id_has_ext = Path(sample_id).suffix.lower() in IMG_EXT
    R.info(f"image_id inclui extensão: {id_has_ext} (ex.: '{sample_id}')")
    missing, ids = {}, {}
    for n, d in dfs.items():
        if n == "classes.csv":
            continue
        ids[n] = [str(v) for v in d[img_col]]
        missing[n] = [i for i in ids[n] if idx.get(i) is None and idx.get(Path(i).stem) is None]
        R.check(not missing[n], f"{n}: todas as imagens existem", f"faltam {len(missing[n])}: {missing[n][:3]}" if missing[n] else "")
    if train_dir is not None:
        fs = sorted(p for p in train_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        ids["train"] = [p.name if id_has_ext else p.stem for p in fs]
        R.info(f"pool: {len(fs)} imagens em {train_dir}")
    else:
        ids["train"] = ids["train.csv"]
    R.check(len(set(ids["train"])) == len(ids["train"]), "ids únicos no pool")
    images_dir = None
    p0 = idx.get(sample_id) or idx.get(Path(sample_id).stem)
    if p0:
        try:
            images_dir = str(p0.parent.relative_to(data))
        except ValueError:
            images_dir = str(p0.parent)

    # ---- contagens do enunciado
    if not a.fake:
        exp = P["tamanhos_esperados"]
        sizes = {"train": len(ids["train"]), "val": len(dfs["val.csv"]), "test_public": len(dfs["test_public.csv"]),
                 **{f"rotulos_{b}": len(dfs[f"rotulos_{b}.csv"]) for b in P["orcamentos"]}}
        for k, v in exp.items():
            R.check(sizes[k] == v, f"tamanho {k}", f"{sizes[k]} (enunciado: {v})")
    # ---- orçamentos
    for b in P["orcamentos"]:
        d = dfs[f"rotulos_{b}.csv"]
        cnt = d.groupby(lab_col).size()
        R.check(len(cnt) == P["n_classes"] and (cnt == b).all(), f"rotulos_{b}: exatamente {b} por classe em {P['n_classes']} classes",
                f"min {cnt.min()} max {cnt.max()}, {len(cnt)} classes")
    s1, s5, s20 = (set(ids[f"rotulos_{b}.csv"]) for b in (1, 5, 20))
    tr = set(ids["train"])
    R.check(s1 <= s5 <= s20, "orçamentos aninhados (1 ⊂ 5 ⊂ 20)")
    R.check(s20 <= tr, "rotulos_* ⊂ train")
    va, te = set(ids["val.csv"]), set(ids["test_public.csv"])
    R.check(not (tr & va) and not (tr & te) and not (va & te), "train, val e test_public disjuntos (por id)")
    # consistência rótulo entre budgets
    m1 = dict(zip(ids["rotulos_5.csv"], dfs["rotulos_5.csv"][lab_col]))
    m2 = dict(zip(ids["rotulos_20.csv"], dfs["rotulos_20.csv"][lab_col]))
    R.check(all(m2[i] == l for i, l in m1.items() if i in m2), "mesmo id => mesmo rótulo em rotulos_5 e rotulos_20")
    vc = dfs["val.csv"].groupby(lab_col).size()
    R.check(vc.max() <= P["val_max_por_classe"], "val: no máx. 10 por classe", f"máx {vc.max()}, classes {len(vc)}", warn_only=a.fake)
    R.check(len(vc) == P["n_classes"], "val cobre todas as classes", f"{len(vc)}/{P['n_classes']}")
    R.check(dfs["test_public.csv"][lab_col].nunique() == P["n_classes"], "test_public cobre todas as classes")

    # ---- cavalo (classe rara)
    horse = [n for n in names_sorted if "horse" in n.lower()]
    if horse and not a.fake:
        key = horse[0] if label_kind == "name" else (int(cdf[cdf[name_col] == horse[0]][id_col_cls].iloc[0]) if id_col_cls else names_sorted.index(horse[0]))
        got = dict(val=int((dfs["val.csv"][lab_col] == key).sum()), test_public=int((dfs["test_public.csv"][lab_col] == key).sum()))
        for k, v in got.items():
            R.check(v == P["cavalo_esperado"][k], f"cavalo no {k}", f"{v} (enunciado: {P['cavalo_esperado'][k]})", warn_only=True)

    # ---- estatísticas das imagens (informam o resize_mode e o size)
    some = ids["train"][: a.max_size_scan]
    dims = np.array([Image.open(idx.get(i) or idx.get(Path(i).stem)).size for i in some])
    longest = dims.max(1)
    R.check(longest.max() <= P["lado_maior_max_px"], "lado maior ≤ 512 px", f"máx {longest.max()}")
    ar = dims[:, 0] / dims[:, 1]
    R.info(f"largura×altura (n={len(dims)}): largura p5/p50/p95 = {np.percentile(dims[:, 0], [5, 50, 95]).astype(int)}, "
           f"altura p5/p50/p95 = {np.percentile(dims[:, 1], [5, 50, 95]).astype(int)}")
    R.info(f"proporção w/h p5/p50/p95 = {np.round(np.percentile(ar, [5, 50, 95]), 2)} -> se variar bastante, vale o sweep de data.resize_mode")
    sat = []
    for i in some[:300]:
        rgb = np.asarray(Image.open(idx.get(i) or idx.get(Path(i).stem)).convert("RGB"), dtype=np.float32)
        sat.append(np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean())
    sat = np.array(sat)
    R.info(f"fração ~sem cor (cor_media_abs_diff < 1) em {len(sat)} imagens: {(sat < 1).mean():.2f} (noturnas/infravermelho)")

    # ---- resumo por classe
    inv = {i: n for i, n in enumerate(names_sorted)}
    def to_name(s):
        return s if label_kind == "name" else s.map(lambda v: inv[id2idx[int(v)]] if id2idx else inv[int(v)])
    id2idx = {int(v): i for i, v in enumerate(cdf.sort_values(id_col_cls)[id_col_cls])} if id_col_cls else {}
    summ = pd.DataFrame({n.replace(".csv", ""): to_name(dfs[n][lab_col]).value_counts()
                         for n in dfs if n not in ("classes.csv", "train.csv")}).reindex(names_sorted).fillna(0).astype(int)
    summ.insert(0, "train_total (nao rotulado)", np.nan)
    out_csv = C.ROOT / "resultados" / "dados_resumo_por_classe.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summ.drop(columns=["train_total (nao rotulado)"]).to_csv(out_csv)
    print("\n" + summ.drop(columns=["train_total (nao rotulado)"]).to_string())

    # ---- schema
    schema = dict(files=files, train_dir=(str(train_dir.relative_to(data)) if train_dir is not None else None),
                  split_cols=dict(image=img_col, label=lab_col), label_kind=label_kind,
                  classes_cols=dict(name=name_col, id=id_col_cls), id_has_extension=bool(id_has_ext),
                  images_dir=images_dir, _detectado_por="tests/check_data.py")
    print("\nschema detectado:", schema)
    if not a.no_write and not any(r[0] == "FAIL" for r in R.rows):
        C.save_json(schema, C.ROOT / "configs" / "data_schema.json")
        R.ok("configs/data_schema.json escrito")
    R.finish()


if __name__ == "__main__":
    main()
