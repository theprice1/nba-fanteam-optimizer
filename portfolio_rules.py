"""Portfolio rules for a set of entries. Checkable without the solver.

Rules (week-1 evidence, revisit each week):
  - any two lineups share at most max_overlap players
  - with 5+ lineups, no player appears in more than max_share of them
  - each lineup has at least min_ceiling players projected at ceiling_proj or more
  - each lineup has at most max_chalk players owned at chalk_own or more
"""
from collections import Counter
import itertools


def check_set(lineups, pool, *, max_overlap=3, max_share=0.4, min_ceiling=2,
              ceiling_proj=35.0, max_chalk=3, chalk_own=30.0):
    """lineups: list of lists of player names. pool: DataFrame indexed by name with proj and own.
    Returns (ok, problems)."""
    problems = []
    n = len(lineups)
    for a, b in itertools.combinations(range(n), 2):
        shared = len(set(lineups[a]) & set(lineups[b]))
        if shared > max_overlap:
            problems.append(f"lineups {a + 1} and {b + 1} share {shared} players (max {max_overlap})")
    if n >= 5:
        cap = int(max_share * n)
        for p, c in Counter(p for lu in lineups for p in lu).items():
            if c > cap:
                problems.append(f"{p} is in {c} of {n} lineups (cap {cap})")
    for k, lu in enumerate(lineups, 1):
        hi = sum(float(pool.loc[p, "proj"]) >= ceiling_proj for p in lu)
        if hi < min_ceiling:
            problems.append(f"lineup {k}: {hi} ceiling player(s), need {min_ceiling}")
        ch = sum(float(pool.loc[p, "own"]) >= chalk_own for p in lu)
        if ch > max_chalk:
            problems.append(f"lineup {k}: {ch} chalk players, max {max_chalk}")
    return (not problems), problems
