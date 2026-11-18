"""Testes unitários (CPU, sem dados, sem internet):  python -m pytest -q tests/test_units.py
ou, sem pytest:  python -m tests.test_units"""
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import common as C  # noqa: E402
from src import fixmatch as FM  # noqa: E402
from src import preproc  # noqa: E402
from src import saliencia as S  # noqa: E402
from src import sweep as SW  # noqa: E402
from src.models import ARCHS, build_model, count_params, load_checkpoint, save_checkpoint, target_layer  # noqa: E402


def _img(w=100, h=50, seed=0):
    return Image.fromarray(np.random.RandomState(seed).randint(0, 255, (h, w, 3), dtype=np.uint8))


def test_resize_shapes_and_pad_center():
    im = _img(100, 50)
    assert preproc.resize_square(im, 64, "squash").size == (64, 64)
    p = np.asarray(preproc.resize_square(im, 64, "pad"))
    assert p.shape == (64, 64, 3)
    fill = np.array([round(m * 255) for m in preproc.MEAN])
    assert (p[0, 0] == fill).all() and (p[-1, -1] == fill).all()      # faixas de preenchimento (imagem larga)
    assert not (p[32, 32] == fill).all()


def test_squash_equals_torchvision_eval_path():
    im = _img(120, 80)
    a = preproc.to_tensor(preproc.resize_square(im, 64, "squash"))
    b = T.Compose([T.Resize((64, 64)), T.ToTensor(), T.Normalize(preproc.MEAN, preproc.STD)])(im)
    assert torch.allclose(a, b, atol=1e-5)


def test_overrides_and_hash():
    cfg = {"a": {"b": 1}, "c": 2, "_x": "ignorado"}
    out = C.apply_overrides(cfg, ["a.b=3", 'd.e="txt"', "c=[1, 2]"])
    assert out["a"]["b"] == 3 and out["d"]["e"] == "txt" and out["c"] == [1, 2] and cfg["a"]["b"] == 1
    assert C.cfg_hash(cfg) == C.cfg_hash({**cfg, "_x": "outro"}) != C.cfg_hash(out)


def test_defaults_are_valid_json_and_complete():
    for m in ("part0_ft", "part0_scratch", "fixmatch"):
        cfg = C.load_json(C.DEFAULTS / f"{m}.json")
        assert {"model", "data", "aug", "train"} <= set(cfg)
        sw = C.load_json(C.ROOT / "configs" / "sweeps" / f"{m}.json")
        for k in sw["space"]:                        # toda chave varrida existe na config
            assert C.cfg_get(cfg, k) is not None, f"{m}: {k} não existe em defaults"


def test_flex_threshold():
    thr = FM.flex_thr(torch.tensor([0.0, 0.5, 1.0]), tau=0.95, floor=0.5)
    assert abs(thr[0].item() - 0.5) < 1e-6                     # piso quando nada foi aprendido
    assert abs(thr[2].item() - 0.95) < 1e-6                    # classe totalmente aprendida: limiar cheio
    assert thr[1].item() < 0.95
    assert abs(FM.flex_thr(torch.tensor([0.0]), tau=0.3, floor=0.5)[0].item() - 0.3) < 1e-6   # nunca acima de tau


def test_distribution_alignment():
    p = torch.softmax(torch.randn(64, 5) + torch.tensor([3.0, 0, 0, 0, 0]), 1)     # modelo viciado na classe 0
    q, pbar = FM.align_dist(p, torch.full((5,), 0.2), m=0.0)                         # m=0: usa só este lote
    assert torch.allclose(q.sum(1), torch.ones(64), atol=1e-5)
    assert q.argmax(1).eq(0).float().mean() < p.argmax(1).eq(0).float().mean()       # reduz o viés para a classe 0
    assert torch.allclose(pbar, p.mean(0))


def test_ema_update():
    m = torch.nn.Linear(2, 2)
    ema = FM.EMA(m, 0.5)
    w0 = ema.m.weight.clone()
    with torch.no_grad():
        m.weight.add_(1.0)
    ema.update(m)
    assert torch.allclose(ema.m.weight, w0 + 0.5, atol=1e-6)
    assert not ema.m.weight.requires_grad


def test_sweep_sampler_reproducible_and_unique():
    space = {"x": {"choices": [1, 2, 3]}, "lr": {"loguniform": [1e-4, 1e-1]}}
    t1, t2 = SW.make_trials(space, 6, 0), SW.make_trials(space, 6, 0)
    assert t1 == t2 and t1[0] == {}
    assert len({str(t) for t in t1}) == len(t1)
    assert all(1e-4 <= t["lr"] <= 1e-1 for t in t1[1:])


def test_curve_csv_is_idempotent():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "c.csv"
        C.append_curve_row(f, "piso", 5, 0, 0.3, 0.4)
        C.append_curve_row(f, "piso", 5, 0, 0.35, 0.45)
        C.append_curve_row(f, "piso", 5, 1, 0.31, 0.41)
        df = pd.read_csv(f)
        assert list(df.columns) == ["metodo", "orcamento", "seed", "f1_macro", "acuracia"]
        assert len(df) == 2 and df[(df.seed == 0)].f1_macro.iloc[0] == 0.35


def test_param_limit_all_archs():
    lim = C.project()["limite_parametros"]
    for a in ARCHS[:3]:
        assert count_params(build_model(a, 21, False)) <= lim, a


def test_checkpoint_roundtrip_and_gradcam():
    names = [f"c{i}" for i in range(4)]
    for arch in ("resnet18", "smallcnn"):
        torch.manual_seed(0)
        m = build_model(arch, 4, False).eval()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.pt"
            save_checkpoint(p, m, arch, names, dict(size=64, resize_mode="pad"), meta={"k": 1})
            m2, ck = load_checkpoint(p)
        x = torch.randn(2, 3, 64, 64)
        assert torch.allclose(m(x), m2(x), atol=1e-6) and ck["class_names"] == names and ck["num_params"] == count_params(m)
        gc = S.GradCAM(m2, target_layer(m2, arch))
        cam, probs, t = gc(x[:1])
        gc.close()
        assert cam.shape == (64, 64) and 0 <= cam.min() and cam.max() <= 1 + 1e-6 and abs(probs.sum() - 1) < 1e-5
        occ, mx = S.occlusion_map(m2, x[:1], t, 16, 8)
        assert occ.shape == (64, 64) and occ.min() >= 0
        st = S.cam_stats(cam, 0.5)
        assert 0 < st["area_frac"] <= 1 and st["bbox_w"] >= 1


def test_unlabeled_views_share_crop():
    """A vista forte parte do MESMO recorte da fraca (senão o animal pequeno some de uma das vistas)."""
    arr = np.stack([np.asarray(_img(80, 80, s)) for s in range(3)])
    aug = dict(weak_scale_min=0.5, randaug_ops=2, randaug_mag=10, erase_p=0.0, erase_scale=[0.02, 0.2])
    ds = FM.UnlDS(arr, 32, aug)
    w, s, i = ds[1]
    assert w.shape == s.shape == (3, 32, 32) and i == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        random.seed(0)
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} testes passaram")
