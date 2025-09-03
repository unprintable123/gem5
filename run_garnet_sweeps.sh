#!/usr/bin/env bash
set -Eeuo pipefail

python run_garnet_sweeps.py \
  --gem5="./build/NULL/gem5.opt" \
  --config="configs/example/garnet_synth_traffic.py" \
  --out-root="logs/runs_rout_cmp" \
  --title="HyperX: packets_received::total vs X (routing algorithm variants)" \
  --base-flags='--network=garnet --num-cpus=64 --num-dirs=64 --topology=HyperX --mesh-rows=4 --synthetic=bit_complement --inj-vnet=0 --vcs-per-vnet=16 --sim-cycles=5000 --dimwar-weight=hop_x_cong' \
  --grid='--routing-algorithm=0,3,4' \
  --label-keys='--routing-algorithm' \
  --x-flag="--injectionrate" \
  --x-range="0.1:0.80:0.1" \
  --x-title='Injection Rate' \
  --y-metric="packets_received::total" \
  --clip="mono" --clip-jump-factor=20 \
  --timeout-sec=900 \
  --max-parallel=1 \
  --save-tag="rout_cmp_quick" \
  --plot \
  --plot-out="logs/runs_rout_cmp/plots/444-BC-vc16" \
  --fig-width=12 \
  --fig-aspect=1.777 \
  --dpi=180 \
  --legend-loc=upper_left \
  --legend-fontsize=18 \
  --legend-compact
