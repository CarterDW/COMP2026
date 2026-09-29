"""Run every rung's checks, in order.

    python checks/run_all.py

New rungs register themselves: any file in this directory named rung*.py that
exposes main() is picked up automatically, so adding a rung means adding one
file and nothing else.
"""

import importlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

HERE = pathlib.Path(__file__).resolve().parent


def main():
    modules = sorted(HERE.glob("rung*.py"))
    failures = sum(importlib.import_module(f"checks.{p.stem}").main() for p in modules)
    print(f"{len(modules)} rung(s) checked: "
          + ("all checks passed." if failures == 0 else f"{failures} check(s) FAILED."))
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
