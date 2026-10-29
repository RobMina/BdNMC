#!/bin/bash
# Stock's reading of a particle list, for tests/test_particle_list.py: run tests/data/plist_formats.dat through a
# stock build in `output_mode particle_list`, which samples the list `samplesize` times and writes each parent as
# `px py pz E x y z t` (E from the pi0 mass; position = the origin read, zero without particle_list_position;
# 6 significant digits). Stock writes to the particle_list_file itself, so it runs on a copy.
# Usage: plist_echo.sh [O2|O0]   writes tests/data/plist_formats_echo_{pos,nopos}.txt
set -eo pipefail
build=${1:-O2}
here=$(cd -P "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
src=$(cd "$root/.." && pwd)
bin=$root/stock/$build/main
[[ -x $bin ]] || { echo "missing build: $bin" >&2; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for pos in pos nopos; do
    cp "$root/tests/data/plist_formats.dat" "$tmp/list_$pos.dat"
    flag=$([[ $pos == pos ]] && echo true || echo false)
    # 20 samples of a 7-row list: the rows in order, then from the first row again
    sed -e "s#^particle_list_file .*#particle_list_file $tmp/list_$pos.dat#" \
        -e "s#^particle_list_position .*#particle_list_position $flag#" \
        -e "s#^output_mode .*#output_mode particle_list#" \
        -e "s#^samplesize .*#samplesize 20#" \
        -e "s#^output_file .*#output_file $tmp/events.dat#" \
        -e "s#^summary_file .*#summary_file $tmp/summary.dat#" "$root/phase0/cards/mb_list_m005.dat" > "$tmp/card_$pos.dat"
    (cd "$src" && DYLD_LIBRARY_PATH="$root/stock/$build" "$bin" "$tmp/card_$pos.dat" > "$tmp/log_$pos.txt" 2>&1)
    cp "$tmp/list_$pos.dat" "$root/tests/data/plist_formats_echo_$pos.txt"
    grep -i "error\|format" "$tmp/log_$pos.txt" || true
    echo "wrote tests/data/plist_formats_echo_$pos.txt ($(wc -l < "$tmp/list_$pos.dat") rows)"
done
