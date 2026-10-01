"""
Parte 4 — classificação dos ID switches da galeria de falhas.

Tipos: morreu (a track antiga morreu antes da volta), nao_recasou (a track
antiga ainda existia e não aceitou a volta), troca (o id da volta já era de
outra pessoa).

Roda com "python tests/test_failure_gallery.py" ou sob pytest.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from failure_gallery import MAX_AGE, WINDOW, classify, diagnose


def event(gap, prev_frame=100, pred_from=15, pred_to=47):
    return {"gap": gap, "prev_frame": prev_frame, "frame": prev_frame + gap,
            "pred_from": pred_from, "pred_to": pred_to, "gt_id": 70,
            "scene": "02", "camera": "static", "density": 31.0}


def test_gap_of_max_age_plus_one_can_still_rematch():
    # Vista no quadro s, a track ainda casa em s + max_age + 1: não morreu.
    spans = {15: (1, 100), 47: (121, 200)}

    assert classify(event(MAX_AGE + 1), spans) == "nao_recasou"
    assert classify(event(MAX_AGE + 2), spans) == "morreu"


def test_old_track_still_emitting_did_not_die():
    # O id antigo continua saindo depois do sumiço (casou com outra pessoa):
    # não morreu, mesmo com buraco longo — é o caso da falha 1 da auditoria.
    spans = {15: (1, 118), 47: (140, 200)}

    assert classify(event(40), spans) == "nao_recasou"


def test_returning_to_an_existing_id_is_a_swap():
    spans = {15: (1, 100), 47: (5, 200)}

    assert classify(event(10), spans) == "troca"


def test_long_gap_diagnosis_mentions_the_training_window():
    # O enunciado liga a falha à janela de BPTT: buraco maior que T nunca
    # teve sinal de supervisão atravessando.
    spans = {15: (1, 100), 47: (140, 200)}
    long_gap = event(WINDOW + 20)
    long_gap["kind"] = classify(long_gap, spans)
    long_gap["pred_from_last"] = spans[15][1]

    assert f"T={WINDOW}" in diagnose(long_gap)


def main():
    tests = [value for name, value in globals().items()
             if name.startswith("test_")]

    for test in tests:
        test()
        print(f"ok  {test.__name__}")


if __name__ == "__main__":
    main()
