#!/bin/bash
# Run the port's V1/V2 shards one after another (one XLA process at a time keeps the laptop under 4 cores).
# Each point: N trials in 4 block-range shards, seed = the stock reference card's seed (independent generator).
# Usage: [V_OUT=dir] [PY=python] [PARENT_SAMPLER=stock|envelope] run_v_set.sh  (outputs in $V_OUT/<stem>/shard<k>.npz,
# default validation/out/v; existing shards are skipped). The backend is JAX's default (GPU when one is visible).
set -o pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
out_dir=${V_OUT:-$root/validation/out/v}
py=${PY:-$root/.venv/bin/python}
sampler=${PARENT_SAMPLER:-stock}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
block=65536
for spec in mb_pi0_m001:500000000 mb_pi0_m005:500000000 mb_pi0_m020:500000000 mb_pi0_m045:500000000 \
            he_pi0_cuboid_m001:40000000 he_pi0_cuboid_m005:40000000 he_pi0_cuboid_m020:40000000 \
            he_pi0_cuboid_m045:40000000; do
    stem=${spec%%:*}; n=${spec##*:}
    card=$root/phase0/cards/ref/$stem.dat
    seed=$(awk '$1 == "seed" {print $2}' "$card")
    nb=$(( (n + block - 1) / block ))
    for k in 0 1 2 3; do
        b0=$(( nb * k / 4 )); b1=$(( nb * (k + 1) / 4 ))
        out=$out_dir/$stem/shard$k.npz
        [[ -e $out ]] && { echo "skip $out"; continue; }
        "$py" "$root/validation/run_port.py" "$card" "$n" "$b0" "$b1" "$out" --seed "$seed" --parent-sampler "$sampler" \
            || echo "FAILED $stem shard$k"
    done
done
echo "all done"
