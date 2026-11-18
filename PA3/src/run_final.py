"""Parte 2 — roda piso e método da trilha nos orçamentos/seeds e preenche resultados/curva_orcamento.csv.

Retomável: run cujo metrics.json já existe é pulado. Usa configs/selected/*.json (escolhidas no val).
O test_public é avaliado uma única vez por run, no checkpoint escolhido pelo val.

  python -m src.run_final --data dados --out resultados
"""
import argparse
from pathlib import Path

from . import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="resultados")
    ap.add_argument("--methods", nargs="+", default=["part0_ft", "fixmatch"])
    ap.add_argument("--budgets", type=int, nargs="+", default=None)
    ap.add_argument("--allow_defaults", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--force", action="store_true", help="refaz runs já existentes")
    ap.add_argument("--set", nargs="*", default=[], metavar="a.b=v")
    a = ap.parse_args()

    P = C.project()
    budgets = a.budgets or P["orcamentos"]
    plan = []
    for b in budgets:
        seeds = P["seeds_orcamento_padrao"] if b == P["orcamento_padrao"] else [P["seeds_orcamento_padrao"][0]]
        for m in a.methods:
            for s in seeds:
                plan.append((m, b, s))
    print("Plano:", *[f"{m} b{b} s{s}" for m, b, s in plan], sep="\n  ")
    cache = C.DataCache(a.data, a.cache)
    for m, b, s in plan:
        d = C.run_dir(a.out, m, b, s)
        if (d / "metrics.json").exists() and not a.force:
            print(f"[pula] {d.name} (já existe)")
            C.append_curve_row(Path(a.out) / "curva_orcamento.csv", P["rotulo_no_csv"][m], b, s,
                               C.load_json(d / "metrics.json")["test_f1"], C.load_json(d / "metrics.json")["test_acc"])
            continue
        cfg = C.method_cfg(m, b, a.set, a.allow_defaults)
        print(f"\n=== {d.name} | config: {cfg['_source']}", flush=True)
        data = cache.get(b, cfg, need_unl=(m == "fixmatch"), need_test=True)
        mod = __import__(f"src.{'fixmatch' if m == 'fixmatch' else 'part0'}", fromlist=["run"])
        kw = dict(method=m)
        res = mod.run(cfg, data, b, s, d, True, verbose=False, **kw)
        C.append_curve_row(Path(a.out) / "curva_orcamento.csv", P["rotulo_no_csv"][m], b, s, res["test_f1"],
                           res["test_acc"])
        print(f"    val F1 {res['val_f1']:.3f} | test F1 {res['test_f1']:.3f} acc {res['test_acc']:.3f} | "
              f"{res['minutes']:.1f} min", flush=True)


if __name__ == "__main__":
    main()
