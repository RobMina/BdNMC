#!/bin/bash
# Run the port's V1/V2 shards one after another (one XLA process at a time keeps the laptop under 4 cores).
# Each point: N trials in 4 block-range shards, seed = the stock reference card's seed (independent generator).
# Usage: [V_SET=pi0|brem] [V_OUT=dir] [PY=python] [PARENT_SAMPLER=stock|envelope] [SEED_OFFSET=n] run_v_set.sh
# (outputs in $V_OUT/<stem>/shard<k>.npz, default validation/out/v; existing shards are skipped). The backend is
# JAX's default (GPU when one is visible). SEED_OFFSET (default 0) is added to every card's seed, for an independent
# repeat. V_SET=brem runs the eight proton-brem points (v_config_brem.json), 2e7 trials each; V_SET=eta the eight
# eta points (v_config_eta.json), 5e8 (mb) and 2e8 (he) trials each; V_SET=list the eight particle_list points
# (v_config_list.json, lists from phase0/make_lists.py), 5e8 and 4e7.
set -o pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
out_dir=${V_OUT:-$root/validation/out/v}
py=${PY:-$root/.venv/bin/python}
sampler=${PARENT_SAMPLER:-stock}
seed_offset=${SEED_OFFSET:-0}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
block=65536
case ${V_SET:-pi0} in
    pi0) specs="mb_pi0_m001:500000000 mb_pi0_m005:500000000 mb_pi0_m020:500000000 mb_pi0_m045:500000000
                he_pi0_cuboid_m001:40000000 he_pi0_cuboid_m005:40000000 he_pi0_cuboid_m020:40000000
                he_pi0_cuboid_m045:40000000" ;;
    brem) specs="mb_brem_m001:20000000 mb_brem_m005:20000000 mb_brem_m020:20000000 mb_brem_m045:20000000
                 he_brem_cuboid_m001:20000000 he_brem_cuboid_m005:20000000 he_brem_cuboid_m020:20000000
                 he_brem_cuboid_m045:20000000" ;;
    eta) specs="mb_eta_m001:500000000 mb_eta_m005:500000000 mb_eta_m020:500000000 mb_eta_m045:500000000
                he_eta_cuboid_m001:200000000 he_eta_cuboid_m005:200000000 he_eta_cuboid_m020:200000000
                he_eta_cuboid_m045:200000000" ;;
    list) specs="mb_list_m001:500000000 mb_list_m005:500000000 mb_list_m020:500000000 mb_list_m045:500000000
                 he_list_cuboid_m001:40000000 he_list_cuboid_m005:40000000 he_list_cuboid_m020:40000000
                 he_list_cuboid_m045:40000000" ;;
    *) echo "unknown V_SET ${V_SET}"; exit 2 ;;
esac
for spec in $specs; do
    stem=${spec%%:*}; n=${spec##*:}
    card=$root/phase0/cards/ref/$stem.dat
    [[ -e $card ]] || card=$root/phase0/cards/$stem.dat  # the 5 MeV brem references are the base cards' runs
    seed=$(( $(awk '$1 == "seed" {print $2}' "$card") + seed_offset ))
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
