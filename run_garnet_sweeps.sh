#!/usr/bin/env bash
set -Eeuo pipefail

# Tip: if this file was edited on Windows, run: dos2unix run_syn_cmp.sh

python run_garnet_sweeps.py \
--gem5="./build/NULL/gem5.opt" \
--config="configs/example/garnet_synth_traffic.py" \
--out-root="runs_syn_cmp" \
--title="HyperX: latency vs X (synthetic variants)" \
--base-flags='--network=garnet --num-cpus=64 --num-dirs=64 --topology=HyperX --mesh-rows=4 --routing-algorithm=3 --inj-vnet=0 --vcs-per-vnet=16 --sim-cycles=5000 --dimwar-weight=cong' \
--grid='--synthetic=uniform_random' \
--label-keys='--synthetic' \
--x-flag="--injectionrate" \
--x-range="0.20:0.40:0.20" \
--x-title='Injection Rate' \
--y-metric="average_packet_latency" \
--clip="mono" --clip-jump-factor=2 \
--timeout-sec=900 \
--max-parallel=1 \
--save-tag="syn_cmp_quick" \
--plot \
--plot-out="plots/syn_cmp_quick" \
--fig-width=12 \
--fig-aspect=1.777 \
--legend-right-frac=0.22 \
--dpi=180 \
--legend-right-frac=0.1 \
--legend-fontsize=4 \
--legend-ncol=2 \
--legend-compact \
--dry-run
