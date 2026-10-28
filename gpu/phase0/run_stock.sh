#!/bin/bash
# Run one Phase 0 card with one stock build; optional macOS `sample` profile.
# Usage: [TAG=name] run_stock.sh <O0|O2> <card-stem> [profile-seconds]  (TAG suffixes the output dir)
set -eo pipefail

build=$1; stem=$2; prof=${3:-0}
here=$(cd -P "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
src=${BDNMC_SRC:-$(cd "$root/.." && pwd)}  # the BdNMC checkout this gpu/ directory lives in (for data/)
bin=$root/stock/$build/main
card=$here/cards/$stem.dat
out=$here/out/$stem/$build${TAG:+_$TAG}

[[ -x $bin ]] || { echo "missing build: $bin" >&2; exit 1; }
[[ -f $card ]] || { echo "missing card: $card" >&2; exit 1; }
mkdir -p "$out"
[[ -e $out/events.dat ]] && { echo "output exists, not overwriting: $out" >&2; exit 1; }

# Redirect the card's outputs into this build's directory.
sed -e "s#^output_file .*#output_file $out/events.dat#" \
    -e "s#^summary_file .*#summary_file $out/summary.dat#" "$card" > "$out/card.dat"

# BdNMC opens data/*.dat relative to the working directory, so run from the source root.
cd "$src"
# SIP strips DYLD_* when launching /usr/bin/time, so env re-sets it for main.
/usr/bin/time -l env DYLD_LIBRARY_PATH="$root/stock/$build" "$bin" "$out/card.dat" > "$out/log.txt" 2> "$out/time.txt" &
pid=$!
if (( prof > 0 )); then
    sleep 5
    sample "$(pgrep -P $pid main || echo $pid)" "$prof" -file "$out/profile.txt" > /dev/null 2>&1 || true
fi
wait $pid
echo "done: $out"
