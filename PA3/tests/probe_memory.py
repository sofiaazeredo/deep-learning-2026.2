"""Mede o pico de memória da GPU do FixMatch com o backbone/tamanho REALMENTE selecionados (o check_pipeline usa os
defaults, que podem ser bem mais leves) e diz qual train.bs cabe.  Memória ∝ bs·(1 + 2·mu) imagens por passo.

  python -m tests.probe_memory --data dados --cache cache
Use o resultado como:  python -m src.sweep --method fixmatch ... --set train.bs=N
"""
import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import common as C  # noqa: E402
from src import fixmatch as FM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--bs", type=int, nargs="+", default=[32, 24, 16, 12, 8], help="candidatos, do maior para o menor")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--margem", type=float, default=0.85, help="usar no máx. esta fração da memória total")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--allow_defaults", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="a.b=v")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("sem GPU: nada a medir (no CPU não há OOM de GPU).")
        return
    cfg = C.method_cfg("fixmatch", a.budget, a.set + [f"train.steps={a.steps}", f"train.eval_every={a.steps}"],
                       a.allow_defaults, use_selected=False)
    total = torch.cuda.get_device_properties(0).total_memory
    data = C.DataCache(a.data, a.cache).get(a.budget, cfg, need_unl=True)
    mu = cfg["train"]["mu"]
    print(f"GPU {torch.cuda.get_device_name(0)} ({total / 1e9:.1f} GB) | backbone {cfg['model']['arch']} | "
          f"size {cfg['data']['size']} | mu {mu} | {cfg['_source']}")
    print(f"{'bs':>4} {'imgs/passo':>11} {'pico GB':>9} {'s/passo':>8}  resultado")
    best = None
    for bs in a.bs:
        c = C.apply_overrides(cfg, {"train.bs": bs})
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        oom, r = False, None
        try:
            r = FM.run(c, data, a.budget, 0, None, False, verbose=False)
        except torch.cuda.OutOfMemoryError:
            oom = True
        peak = torch.cuda.max_memory_allocated() / 1e9
        gc.collect(); torch.cuda.empty_cache()
        if oom:
            print(f"{bs:>4} {bs * (1 + 2 * mu):>11} {'—':>9} {'—':>8}  OOM")
            continue
        ok = peak <= a.margem * total / 1e9
        print(f"{bs:>4} {bs * (1 + 2 * mu):>11} {peak:>9.2f} {r['sec_per_step']:>8.2f}  {'cabe' if ok else 'cabe, mas sem margem'}")
        if ok and best is None:
            best = bs
    if best:
        print(f"\nRecomendado: --set train.bs={best}  (maior lote com pico ≤ {a.margem:.0%} da memória).")
        print("Se bs ficar muito menor que 32, considere também reduzir data.size no sweep do piso.")
    else:
        print("\nNenhum lote coube. Reduza train.mu, o data.size ou escolha um backbone mais leve no sweep do piso.")


if __name__ == "__main__":
    main()
