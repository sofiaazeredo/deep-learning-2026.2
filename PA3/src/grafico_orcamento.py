"""Parte 2 — curva de orçamento (F1 macro × nº de rótulos, eixo x em log, com a linha do teto) e ganho por classe.

  python -m src.grafico_orcamento --out resultados
Entradas: resultados/curva_orcamento.csv e resultados/runs/*/f1_por_classe.csv. O teto vem de configs/project.json.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="resultados")
    ap.add_argument("--metodo", default="fixmatch", help="método da trilha (nome em project.rotulo_no_csv)")
    ap.add_argument("--teto", type=float, default=None)
    a = ap.parse_args()
    P = C.project()
    out = Path(a.out)
    dest = out / "parte2"
    dest.mkdir(parents=True, exist_ok=True)
    teto = a.teto if a.teto is not None else P["teto_f1_test_public"]
    n_classes = P["n_classes"]
    df = pd.read_csv(out / "curva_orcamento.csv")
    df["n_rotulos"] = df.orcamento * n_classes

    tab = df.groupby(["metodo", "orcamento"]).agg(n_seeds=("seed", "nunique"), f1_media=("f1_macro", "mean"),
                                                  f1_desvio=("f1_macro", "std"), acc_media=("acuracia", "mean"),
                                                  acc_desvio=("acuracia", "std")).reset_index()
    tab.to_csv(dest / "tabela_parte2.csv", index=False)
    print(tab.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for m, g in tab.groupby("metodo"):
        x = g.orcamento * n_classes
        ax.errorbar(x, g.f1_media, yerr=g.f1_desvio.fillna(0), marker="o", capsize=3, label=m)
    ax.axhline(teto, color="k", ls="--", label=f"teto (todos os rótulos) = {teto:.3f}")
    ax.set_xscale("log")
    ax.set_xticks(sorted(df.n_rotulos.unique()))
    ax.set_xticklabels([f"{n}\n({n // n_classes}/classe)" for n in sorted(df.n_rotulos.unique())])
    ax.set(xlabel="número de rótulos (escala log)", ylabel="F1 macro no test_public",
           title="Curva de orçamento (barra = desvio entre seeds; só no orçamento 5 há 3 seeds)")
    ax.title.set_fontsize(9)
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "grafico_orcamento.png", dpi=170)
    plt.close(fig)

    # ganho por classe no orçamento padrão
    b = P["orcamento_padrao"]
    piso_m = [k for k, v in P["rotulo_no_csv"].items() if v == "piso"][0]

    def mean_f1(method):
        fs = sorted((out / "runs").glob(f"{method}_b{b}_s*/f1_por_classe.csv"))
        if not fs:
            return None, 0
        return pd.concat([pd.read_csv(f).set_index("classe") for f in fs]).groupby("classe", sort=False).mean(), len(fs)

    pi, n1 = mean_f1(piso_m)
    me, n2 = mean_f1(a.metodo)
    if pi is None or me is None:
        print("sem f1_por_classe para o ganho por classe (rode src.run_final antes)")
        return
    g = pd.DataFrame(dict(n_test=me.n_test, f1_piso=pi.f1, f1_metodo=me.f1, ganho=me.f1 - pi.f1)).sort_values("ganho")
    g.to_csv(dest / f"ganho_por_classe_{a.metodo}.csv")
    print(f"\nGanho por classe no orçamento {b} (média de {n2} seeds do método vs {n1} do piso):")
    print(g.sort_values("ganho", ascending=False).round(3).to_string())
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["tab:green" if v > 0 else ("tab:red" if v < 0 else "gray") for v in g.ganho]
    ax.barh([f"{c} (n={int(n)})" for c, n in zip(g.index, g.n_test)], g.ganho, color=colors)
    ax.axvline(0, color="k", lw=.8)
    ax.set(xlabel=f"Δ F1 ({a.metodo} − piso), orçamento {b}", title="Ganho por classe (n = imagens no test_public)")
    fig.tight_layout()
    fig.savefig(dest / f"ganho_por_classe_{a.metodo}.png", dpi=170)
    plt.close(fig)
    print(f"\nSalvo em {dest}/")


if __name__ == "__main__":
    main()
