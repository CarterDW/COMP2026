#!/usr/bin/env bash
# Start a formation run detached from this terminal/session (survives closing the app; sleep only pauses it).
# Usage: ./launch.sh TAG [run_formation.py options...]     e.g.  ./launch.sh peb_s1 --pebbles --seed 1 --resume
# Log: data/TAG.log.  Re-running the same command with --resume continues from the last checkpoint.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
tag="$1"; shift
setsid nohup python -u run_formation.py --tag "$tag" "$@" > "data/$tag.log" 2>&1 < /dev/null &
echo "started $tag (pid $!), log: $(pwd)/data/$tag.log"
