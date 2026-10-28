#!/bin/bash
# Run a command and sample its %CPU (sum over threads, 100 = one core) every second; print max and mean.
# Usage: cpu_monitor.sh <command...>
set -o pipefail
"$@" &
pid=$!
max=0; sum=0; n=0
while kill -0 $pid 2>/dev/null; do
    c=$(ps -o %cpu= -p $pid 2>/dev/null | tr -d ' ')
    if [[ -n $c ]]; then
        n=$((n + 1)); sum=$(echo "$sum + $c" | bc)
        if (( $(echo "$c > $max" | bc) )); then max=$c; fi
    fi
    sleep 1
done
wait $pid
rc=$?
(( n > 0 )) && echo "cpu% samples=$n max=$max mean=$(echo "scale=1; $sum / $n" | bc)"
exit $rc
