#!/usr/bin/env bash
# Start terrestrial runs detached (they survive closing the app; sleep only pauses them).
# Usage: ./launch.sh SEED [SEED ...] [-- run_terrestrial.py options]   e.g.  ./launch.sh 1 2 3 -- --planetesimals 200
# Logs: data/terrestrial_seedN[_plM].log, appended to, so a --resume continues the same log.
# Pause: ./pause.sh (keeps the last checkpoint); continue with the same launch command plus --resume.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
seeds=(); while [[ $# -gt 0 && "$1" != "--" ]]; do seeds+=("$1"); shift; done
[[ $# -gt 0 ]] && shift
suffix=""; args=("$@")
for ((k = 0; k < ${#args[@]}; k++)); do
    [[ "${args[k]}" == "--planetesimals" && "${args[k+1]}" != "0" ]] && suffix="_pl${args[k+1]}"
done
for s in "${seeds[@]}"; do
    log="data/terrestrial_seed$s$suffix.log"
    setsid nohup python -u run_terrestrial.py --seed "$s" "$@" >> "$log" 2>&1 < /dev/null &
    echo "started seed $s (pid $!), log: $(pwd)/$log"
done
