"""
Parte 5 — campo receptivo teórico (slides 35-38).

Calcula o campo receptivo camada a camada das arquiteturas da ablação e
compara com a distribuição de tamanhos dos núcleos do DSB2018.

O PDF exige explicitamente: "calculem o campo receptivo teórico do seu
encoder e comparem com a distribuição de tamanhos dos objetos do
dataset. Se usaram atrous convolution, mostrem o campo receptivo com e
sem ela, para a mesma resolução de saída."

Recorrências usadas (slides 35-38):

    conv k, dilatação d:  k_eff = k + (k - 1)(d - 1)
                          rf   += (k_eff - 1) * jump
    max-pool 2x2 s=2:     rf   += jump ; jump *= 2
    upsample 2x:          jump /= 2

A ASPP tem ramos paralelos, então o campo receptivo da saída do bloco é
o MAIOR entre os ramos — é o ramo de maior dilatação que manda.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import DSB2018Dataset


DATA_ROOT = "data/raw"
FIGURES_DIR = Path("experiments/figures/receptive_field")
RESULTS_DIR = Path("experiments/results")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 = todas as 670 amostras.",
    )

    return parser.parse_args()


class ReceptiveField:
    """
    Acumula campo receptivo e jump ao longo de uma pilha de camadas.
    """

    def __init__(self):
        self.rf = 1
        self.jump = 1
        self.trace = []

    def conv(self, k=3, dilation=1, label=None):

        k_eff = k + (k - 1) * (dilation - 1)

        self.rf += (k_eff - 1) * self.jump

        if label:
            self.record(label)

        return self

    def pool(self, label=None):

        self.rf += self.jump
        self.jump *= 2

        if label:
            self.record(label)

        return self

    def upsample(self, label=None):

        self.jump = max(self.jump // 2, 1)

        if label:
            self.record(label)

        return self

    def record(self, label):
        self.trace.append((label, self.rf, self.jump))
        return self

    def report(self, title):

        print()
        print(title)
        print("-" * len(title))

        for label, rf, jump in self.trace:
            print(f"  {label:<28} RF={rf:>5}  jump={jump}")

        print(f"  {'FINAL':<28} RF={self.rf:>5}  jump={self.jump}")


def unet_receptive_field(with_aspp=False):
    """
    Campo receptivo da U-Net do src/model.py.

    4 níveis de encoder (DoubleConv + pool), bottleneck (DoubleConv),
    4 níveis de decoder (upsample + DoubleConv). Com ASPP, o bloco
    entra depois do bottleneck, na MESMA resolução de saída — que é o
    que torna a comparação com/sem atrous justa.
    """

    field = ReceptiveField()

    for level in range(1, 5):
        field.conv(3).conv(3, label=f"encoder {level} (2x conv3)")
        field.pool(label=f"encoder {level} (pool 2x2)")

    field.conv(3).conv(3, label="bottleneck (2x conv3)")

    if with_aspp:
        # Ramos paralelos 1x1, 3x3 d=2, 3x3 d=4, 3x3 d=8.
        # O campo receptivo do bloco é o do ramo mais dilatado.
        before = field.rf

        field.conv(3, dilation=8, label="ASPP (ramo d=8, o maior)")

        print(
            f"  [ASPP soma {field.rf - before} px de campo receptivo "
            f"na mesma resolução]"
        )

    for level in range(4, 0, -1):
        field.upsample(label=f"decoder {level} (upsample 2x)")
        field.conv(3).conv(3, label=f"decoder {level} (2x conv3)")

    return field


def nuclei_diameters(max_samples=0):
    """
    Diâmetro equivalente de cada núcleo, em pixels, NA RESOLUÇÃO EM QUE
    O MODELO ENXERGA (o dataset redimensiona tudo para 256x256).
    """

    dataset = DSB2018Dataset(DATA_ROOT)

    total = len(dataset)

    if max_samples:
        total = min(total, max_samples)

    diameters = []

    for index in range(total):

        labels = dataset[index]["instance_mask"].numpy()

        ids, areas = np.unique(labels, return_counts=True)

        for instance_id, area in zip(ids, areas):

            if instance_id == 0:
                continue

            diameters.append(
                2.0 * np.sqrt(area / np.pi)
            )

    return np.array(diameters)


def main():

    args = parse_args()

    print("=" * 66)
    print("PARTE 5 — CAMPO RECEPTIVO TEÓRICO")
    print("=" * 66)

    plain = unet_receptive_field(with_aspp=False)
    plain.report("U-Net (sem atrous)")

    aspp = unet_receptive_field(with_aspp=True)
    aspp.report("U-Net + ASPP (com atrous, mesma resolução de saída)")

    print()
    print(f"Campo receptivo sem ASPP: {plain.rf} px")
    print(f"Campo receptivo com ASPP: {aspp.rf} px "
          f"(+{aspp.rf - plain.rf} px, sem mudar a resolução)")

    # ----------------------------------------------------------------
    # Distribuição de tamanhos
    # ----------------------------------------------------------------

    print()
    print("Medindo os núcleos do dataset (256x256, como o modelo vê)...")

    diameters = nuclei_diameters(args.max_samples)

    percentiles = [50, 90, 99, 100]
    values = np.percentile(diameters, percentiles)

    print()
    print(f"{len(diameters)} núcleos medidos")

    for p, v in zip(percentiles, values):
        print(f"  p{p:<3} diâmetro equivalente = {v:6.1f} px")

    print(f"  média                      = {diameters.mean():6.1f} px")

    larger_plain = (diameters > plain.rf).mean() * 100
    larger_aspp = (diameters > aspp.rf).mean() * 100

    print()
    print(f"Núcleos maiores que o campo receptivo sem ASPP: "
          f"{larger_plain:.2f}%")
    print(f"Núcleos maiores que o campo receptivo com ASPP: "
          f"{larger_aspp:.2f}%")

    print()

    if larger_plain < 1.0:
        print(
            "LEITURA: o campo receptivo NÃO é o gargalo deste dataset. "
            "Praticamente todo núcleo cabe folgadamente dentro dele, "
            "então as falhas da Parte 5 precisam de outro diagnóstico "
            "(perda de resolução no encoder, fronteira entre núcleos "
            "encostados, contraste) e não de mais contexto."
        )
    else:
        print(
            f"LEITURA: {larger_plain:.1f}% dos núcleos excedem o campo "
            f"receptivo — para esses, o pixel central nunca enxerga as "
            f"duas bordas do objeto."
        )

    # ----------------------------------------------------------------
    # Figura
    # ----------------------------------------------------------------

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(diameters, bins=60, color="#4C78A8", edgecolor="white")

    ax.axvline(
        plain.rf,
        color="#E45756",
        linestyle="--",
        linewidth=2,
        label=f"RF sem ASPP = {plain.rf} px",
    )

    ax.axvline(
        aspp.rf,
        color="#F58518",
        linestyle="--",
        linewidth=2,
        label=f"RF com ASPP = {aspp.rf} px",
    )

    ax.axvline(
        np.median(diameters),
        color="#54A24B",
        linewidth=2,
        label=f"mediana dos núcleos = {np.median(diameters):.0f} px",
    )

    ax.set_xlabel("diâmetro equivalente do núcleo (px, em 256x256)")
    ax.set_ylabel("número de núcleos")
    ax.set_title(
        "Tamanho dos núcleos x campo receptivo teórico"
    )
    ax.legend()

    fig.tight_layout()

    figure_path = FIGURES_DIR / "campo_receptivo_vs_tamanhos.png"
    fig.savefig(figure_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ----------------------------------------------------------------
    # CSV
    # ----------------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    import csv

    csv_path = RESULTS_DIR / "receptive_field.csv"

    with open(csv_path, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(["metric", "value"])
        writer.writerow(["rf_sem_aspp_px", plain.rf])
        writer.writerow(["rf_com_aspp_px", aspp.rf])
        writer.writerow(["n_nucleos", len(diameters)])
        writer.writerow(["diametro_mediano_px", float(np.median(diameters))])
        writer.writerow(["diametro_p99_px", float(np.percentile(diameters, 99))])
        writer.writerow(["diametro_max_px", float(diameters.max())])
        writer.writerow(["pct_maiores_que_rf_sem_aspp", float(larger_plain)])
        writer.writerow(["pct_maiores_que_rf_com_aspp", float(larger_aspp)])

    print(f"Salvo: {figure_path}")
    print(f"Salvo: {csv_path}")


if __name__ == "__main__":
    main()
