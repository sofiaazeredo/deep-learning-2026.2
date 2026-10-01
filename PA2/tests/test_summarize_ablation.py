"""
Parte 3, Eixo 3 — tabelas de sobrevivência da identidade através da oclusão.

A pergunta do eixo ("qual entrada sustenta a identidade numa oclusão longa, e
isso muda com a densidade?") é respondida por survival_tables. Os testes
montam uma oclusão conhecida e conferem que ela cai no split certo, com o n
certo.

Roda sob pytest (usa tmp_path).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.dataset import SCENES
from summarize_ablation import survival_tables


def _scene(occluded):
    """
    Um pedestre por 30 quadros; com `occluded`, visibility 0,1 nos quadros
    10 a 19 (uma oclusão de 10 quadros).
    """

    gt = []
    for frame in range(1, 31):
        vis = 0.1 if occluded and 10 <= frame <= 19 else 1.0
        gt.append((frame, 1, 10.0 + frame, 20.0, 10.0, 30.0, 1, 1, vis))
    info = {"gt_all": gt, "density": 1.0}
    return info, gt


def _tracks_csv(path, scenes_with_id):
    lines = ["sequence,frame,id,x,y,w,h"]
    for scene, same_id in scenes_with_id.items():
        for frame in range(1, 31):
            if 10 <= frame <= 19:
                continue                           # escondido: sem caixa
            identity = 7 if (frame < 10 or same_id) else 8
            lines.append(f"{scene},{frame},{identity},{10.0 + frame},20.0,10.0,30.0")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_occlusion_lands_in_its_split_with_its_n(tmp_path):
    # Oclusão só na 09 (teste): aparece em "teste" e em "todas", nunca em
    # "treino"; o id sobrevive (mesmo id previsto antes e depois).
    sequences = {s: _scene(occluded=(s == "09")) for s in SCENES}
    path = _tracks_csv(tmp_path / "run.csv", {s: True for s in SCENES})

    by_length, _ = survival_tables([("appearance", 42, str(path))], sequences)

    row = by_length[(by_length["bucket"] == "6-20")].set_index("split")
    assert row.loc["teste", "survival_mean"] == 1.0
    assert row.loc["teste", "n_mean"] == 1
    assert row.loc["todas", "n_mean"] == 1
    assert row.loc["treino", "n_mean"] == 0


def test_new_id_after_the_occlusion_does_not_survive(tmp_path):
    sequences = {s: _scene(occluded=(s == "09")) for s in SCENES}
    path = _tracks_csv(tmp_path / "run.csv", {s: False for s in SCENES})

    by_length, _ = survival_tables([("appearance", 42, str(path))], sequences)

    row = by_length[(by_length["bucket"] == "6-20") & (by_length["split"] == "teste")]
    assert row["survival_mean"].iloc[0] == 0.0
