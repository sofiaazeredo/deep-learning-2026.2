#!/usr/bin/env bash
# Reproduz TUDO a partir do pacote de dados. Retomável: rode de novo se o Colab cair.
#   bash scripts/reproduzir.sh dados resultados
# Variáveis (tamanho dos sweeps; meça o tempo com tests.check_pipeline antes de escolher):
#   N_FT=20 N_SCRATCH=12 N_FM=8 FM_STEPS_SWEEP=1500 SWEEP_SEEDS="0 1" CACHE=/content/cache
set -euo pipefail
DADOS=${1:-dados}; OUT=${2:-resultados}
N_FT=${N_FT:-20}; N_SCRATCH=${N_SCRATCH:-12}; N_FM=${N_FM:-8}
FM_STEPS_SWEEP=${FM_STEPS_SWEEP:-1500}; SWEEP_SEEDS=${SWEEP_SEEDS:-"0 1"}; CACHE=${CACHE:-cache}

python -m tests.test_units
python -m tests.check_data --data "$DADOS"

# 1) hiperparâmetros escolhidos SÓ no val -> configs/selected/*.json
python -m src.sweep --data "$DADOS" --method part0_scratch --budget 5 --seeds $SWEEP_SEEDS --n_trials "$N_SCRATCH" --out "$OUT/sweeps" --cache "$CACHE"
python -m src.sweep --data "$DADOS" --method part0_ft      --budget 5 --seeds $SWEEP_SEEDS --n_trials "$N_FT"      --out "$OUT/sweeps" --cache "$CACHE"
for b in 1 20; do
  python -m src.sweep --data "$DADOS" --method part0_ft --budget $b --seeds 0 --n_trials "$N_SCRATCH" --out "$OUT/sweeps" --cache "$CACHE"
done
python -m src.sweep --data "$DADOS" --method fixmatch --budget 5 --seeds 0 --n_trials "$N_FM" --out "$OUT/sweeps" --cache "$CACHE" --set train.steps="$FM_STEPS_SWEEP"

# 2) Parte 0 + Parte 2 (piso e FixMatch; 3 seeds no orçamento 5) -> resultados/curva_orcamento.csv
python -m src.part0 --data "$DADOS" --out "$OUT" --method both --budget 5 --seed 0 --cache "$CACHE"
python -m src.run_final --data "$DADOS" --out "$OUT" --cache "$CACHE"
python -m src.grafico_orcamento --out "$OUT"

# 3) modelo entregue (escolhido no val), Parte 3 e teste do predict.py
python -m src.select_final --out "$OUT" --method fixmatch --data "$DADOS" --dest pesos/modelo_final.pt
python -m src.galeria --data "$DADOS" --ckpt pesos/modelo_final.pt --out "$OUT/parte3"
python -m tests.check_predict --data "$DADOS" --ckpt pesos/modelo_final.pt
