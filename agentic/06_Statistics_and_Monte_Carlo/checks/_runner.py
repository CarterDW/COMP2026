"""Shared reporting for the per-rung check files.

Each check is a zero-argument function returning (ok, detail): a bool, and a
one-line string quoting the number that was actually compared. The detail string
is printed even when the check passes -- a check that only prints on failure
hides how much margin it had, and the margin is usually the interesting part.
"""


def run(title, checks):
    """Run a list of checks, print a line each, and return the number that failed."""
    print(f"--- {title} ---")
    failures = 0
    for check in checks:
        ok, detail = check()
        print(f"[{'PASS' if ok else 'FAIL'}] {check.__name__}\n         {detail}")
        failures += not ok
    print(f"{len(checks) - failures}/{len(checks)} passed.\n")
    return failures
