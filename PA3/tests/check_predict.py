"""Testa o predict.py como o monitor vai rodar: CPU, sem internet, CSV image_id,label, tempo.

  python -m tests.check_predict --data dados --ckpt pesos/modelo_final.pt
  python -m tests.check_predict --data /content/fake --fake --make_dummy --set model.pretrained=false data.size=64
(--make_dummy cria um checkpoint de pesos aleatórios só para validar o encanamento.)
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import common as C  # noqa: E402
from src.models import build_model, load_checkpoint, save_checkpoint  # noqa: E402
from tests._report import Report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default=str(C.ROOT / "pesos" / "modelo_final.pt"))
    ap.add_argument("--make_dummy", action="store_true")
    ap.add_argument("--n", type=int, default=100, help="nº de imagens do val para o teste")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="a.b=v")
    a = ap.parse_args()
    R = Report("check_predict")
    P = C.project()
    schema = C.load_schema()
    names, id2idx = C.load_classes(a.data, schema)
    tmp = Path(tempfile.mkdtemp())

    ckpt = a.ckpt
    if a.make_dummy:
        cfg = C.method_cfg("part0_ft", 5, a.set, allow_defaults=True)
        cfg["model"]["pretrained"] = False
        model = build_model(cfg["model"]["arch"], len(names), False)
        ckpt = str(tmp / "dummy.pt")
        save_checkpoint(ckpt, model, cfg["model"]["arch"], names,
                        dict(size=cfg["data"]["size"], resize_mode=cfg["data"]["resize_mode"]),
                        meta=dict(id_format="filename" if schema["id_has_extension"] else "stem"))
        R.warn("usando checkpoint DUMMY (pesos aleatórios)")
    model, ck = load_checkpoint(ckpt)
    R.check(ck["num_params"] <= P["limite_parametros"], "parâmetros ≤ 12M", f"{ck['num_params']:,}")
    R.check(ck["class_names"] == names, "class_names do checkpoint == classes.csv")

    ids, paths, y = C.read_split(a.data, schema, "val", names, id2idx)
    n = min(a.n, len(ids))
    folder = tmp / "imgs"
    folder.mkdir()
    for p in paths[:n]:
        os.symlink(Path(p).resolve(), folder / Path(p).name)
    out_csv = tmp / "preds.csv"
    torch_home = tmp / "torch_home"
    torch_home.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TORCH_HOME=str(torch_home),
               HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9", http_proxy="http://127.0.0.1:9",
               https_proxy="http://127.0.0.1:9")           # proxy inválido: qualquer tentativa de rede falha
    r = subprocess.run([sys.executable, str(C.ROOT / "predict.py"), "--imagens", str(folder), "--saida", str(out_csv),
                        "--pesos", ckpt], capture_output=True, text=True, env=env)
    R.check(r.returncode == 0, "predict.py roda (CPU, rede bloqueada)", r.stderr.strip().splitlines()[-1] if r.returncode else "")
    if r.returncode:
        print(r.stdout, r.stderr)
        R.finish()
    print(r.stdout.strip())
    R.check(not any(torch_home.iterdir()), "nada foi baixado (TORCH_HOME vazio)")
    pn = [int(l.split(":")[1]) for l in r.stdout.splitlines() if l.startswith("parametros")]
    R.check(pn and pn[0] == ck["num_params"], "imprime nº de parâmetros", f"{pn}")

    df = pd.read_csv(out_csv, dtype=str)
    R.check(list(df.columns) == ["image_id", "label"], "colunas exatamente image_id,label", str(list(df.columns)))
    R.check(len(df) == n, "uma linha por imagem", f"{len(df)}/{n}")
    R.check(set(df.label) <= set(names), "labels são nomes de classes.csv")
    R.check(set(df.image_id) == set(ids[:n]), "image_id no mesmo formato dos CSVs do pacote",
            f"ex.: predito '{df.image_id.iloc[0]}' vs csv '{ids[0]}'")

    # equivalência com o caminho de TREINO/AVALIAÇÃO (arrays pré-redimensionados + EvalTF)
    pre = ck["preprocess"]
    arr = C.build_array(paths[:n], pre["size"], pre["resize_mode"])
    ev = C.evaluate(model, C.make_loader(arr, y[:n], C.EvalTF(), 64, False), "cpu")
    pred_train_path = np.array([names[i] for i in ev["preds"]])
    got = df.set_index("image_id").loc[ids[:n], "label"].values
    agree = float((pred_train_path == got).mean())
    R.check(agree >= 0.99, "predict.py == caminho de avaliação do treino", f"concordância {agree:.3f}")

    import re
    m = re.search(r"em ([\d.]+)s", r.stdout)
    if m:
        per = float(m.group(1)) / n
        est = per * P["test_private_aprox"] / 60
        R.check(est < P["predict_tempo_max_min"], f"~{P['test_private_aprox']} imagens < {P['predict_tempo_max_min']} min em CPU",
                f"estimativa {est:.1f} min nesta máquina ({per * 1000:.0f} ms/img, inclui carga do modelo)", warn_only=True)
    R.finish()


if __name__ == "__main__":
    main()
