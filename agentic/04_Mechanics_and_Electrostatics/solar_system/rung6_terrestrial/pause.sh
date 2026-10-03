#!/usr/bin/env bash
# Stop every running terrestrial run. Each keeps its last checkpoint (written every 100 kyr, atomically), so at most
# ~100 kyr of work is lost; continue with ./launch.sh SEEDS -- OPTIONS --resume.
# (To pause without losing anything, while the machine stays up: pkill -STOP -f run_terrestrial.py, and -CONT to go on.)
set -euo pipefail
pkill -f "python -u run_terrestrial.py" && echo "stopped" || echo "no runs were running"
