#!/usr/bin/env bash
# Start terrestrial runs detached (they survive closing the app; sleep only pauses them).
# Usage: ./launch.sh SEED [SEED ...] [-- run_terrestrial.py options]     e.g.  ./launch.sh 1 2 3 4 5 6 7 8 -- --resume
# Logs: data/terrestrial_seedN.log. Re-running with --resume continues each from its last checkpoint.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
seeds=(); while [[ $# -gt 0 && "$1" != "--" ]]; do seeds+=("$1"); shift; done
[[ $# -gt 0 ]] && shift
for s in "${seeds[@]}"; do
    setsid nohup python -u run_terrestrial.py --seed "$s" "$@" > "data/terrestrial_seed$s.log" 2>&1 < /dev/null &
    echo "started seed $s (pid $!), log: $(pwd)/data/terrestrial_seed$s.log"
done
