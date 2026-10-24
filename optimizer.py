"""Courtside core: load Fanteam and projection data, value players, solve lineups."""

import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import pulp

SLOTS = ["PG", "SG", "SF", "PF", "C"]
POS_MAP = {
    "guard_point": "PG",
    "guard_shooting": "SG",
    "forward_small": "SF",
    "forward_power": "PF",
    "center": "C",
}
AVAIL = {
    "confirmed": 1.0,
    "expected": 1.0,
    "possible": 0.75,
    "bench": 0.45,
    "unexpected": 0.10,
    "injured": 0.0,
    "refuted": 0.0,
}
TEAM_ALIAS = {
    "SAS": "SA",
    "NOR": "NOP",
    "NO": "NOP",
    "GS": "GSW",
    "PHX": "PHO",
    "UTAH": "UTA",
    "BRK": "BKN",
    "WSH": "WAS",
    "CHO": "CHA",
}
STACK_MIN = 2
MODE_CASH = ("Cash", "Pure projection")
# Auto rotates stack type and ownership penalty so the set covers different builds.
SCHEDULE = [
    ("team", 0.10),
    ("bring-back", 0.10),
    ("none", 0.15),
    ("team", 0.05),
    ("bring-back", 0.15),
    ("none", 0.05),
    ("team", 0.10),
    ("bring-back", 0.05),
    ("none", 0.10),
    ("team", 0.15),
]


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def team_code(s):
    s = str(s).strip().upper()
    return TEAM_ALIAS.get(s, s)


def _read(f):
    if hasattr(f, "seek"):
        f.seek(0)
    df = pd.read_csv(f, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def _col(df, *names):
    lower = {c.lower(): c for c in df.columns}
    return next((lower[n] for n in names if n in lower), None)


def _best_mate(pool, pid, team, pos):
    """Best projection among available same-team, same-position teammates."""
    m = pool.loc[
        (pool.team == team) & (pool.pos == pos) & (pool.pid != pid) & (pool.p > 0),
        "proj",
    ]
    return float(m.max()) if m.notna().any() else 0.0


def load_players(fan_file, dft_file, sharpness):
    """Fanteam list is the authority for ID, position, price and availability.
    Projections come from the Daily Fantasy Tips CSV, matched by Fanteam ID or name."""
    fp = _read(fan_file)
    tc_id = _col(fp, "tournament", "tournamentid", "tournament_id")
    tid_series = fp[tc_id].str.strip() if tc_id else ""

    pool = pd.DataFrame(
        {
            "tid": tid_series,
            "pid": fp["PlayerID"].str.strip(),
            "name": fp["FName"].str.strip() + " " + fp["Name"].str.strip(),
            "team": fp["Club"].map(team_code),
            "pos": fp["Position"].str.strip().map(POS_MAP),
            "status": fp["Lineup"].str.strip().str.lower(),
            "price": pd.to_numeric(fp["Price"], errors="coerce"),
        }
    )
    pool["key"] = pool["name"].map(norm)
    pool["p"] = pool["status"].map(AVAIL).fillna(0.0)
    pool["proj"] = np.nan
    pool["opp"] = None

    dp = _read(dft_file)
    pc = _col(dp, "proj", "projection")
    if pc is None:
        raise ValueError("Projections file needs a Proj column")
    dp["_proj"] = pd.to_numeric(dp[pc], errors="coerce")
    ic = _col(dp, "fanteam id", "fanteam_id", "playerid", "player id")
    nc = _col(dp, "player", "name")
    tc, oc = _col(dp, "team"), _col(dp, "opp", "opponent")
    if ic:
        pool["proj"] = pool["pid"].map(dp.drop_duplicates(ic).set_index(ic)["_proj"])
    if nc:
        dp["_key"] = dp[nc].map(norm)
        byname = dp.drop_duplicates("_key").set_index("_key")["_proj"]
        pool["proj"] = pool["proj"].fillna(pool["key"].map(byname))
    if tc and oc:
        opp = {
            team_code(t): team_code(o)
            for t, o in zip(dp[tc], dp[oc])
            if isinstance(o, str)
        }
        pool["opp"] = pool["team"].map(opp)

    # Safety net: if a player is out, the best healthy teammate at his position takes his minutes.
    pool["safety"] = [
        _best_mate(pool, i, t, s) for i, t, s in zip(pool.pid, pool.team, pool.pos)
    ]
    pool["ev"] = pool.p * pool.proj + (1 - pool.p) * pool.safety
    pool = pool[
        (pool.p > 0) & pool.proj.notna() & pool.price.notna() & pool.pos.notna()
    ].copy()

    # Projected ownership prior (sums to 700% = 7 slots x 100%); replace with history later.
    raw = np.exp(sharpness * pool.proj.clip(lower=0)) * pool.p
    pool["own"] = (700 * raw / raw.sum()).clip(upper=100)
    pool["sigma"] = np.maximum(3.0, 0.35 * pool.ev)
    return pool.reset_index(drop=True)


def _value(pool, mode, lam):
    if mode in MODE_CASH:
        return pool.ev.to_numpy()
    # Tournament value: expected points, rewarded for upside, penalised for crowd ownership.
    return (pool.ev + 0.3 * pool.sigma - lam * pool.own).to_numpy()


def solve_one(pool, value, budget, stack, max_team, overlap, prior, locked, opp_of):
    ids = pool.pid.tolist()
    price = dict(zip(ids, pool.price))
    pos = dict(zip(ids, pool.pos))
    by_team = pool.groupby("team").pid.apply(list).to_dict()

    prob = pulp.LpProblem("lineup", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", ids, cat="Binary")
    prob += pulp.lpSum(float(v) * x[i] for v, i in zip(value, ids))
    prob += pulp.lpSum(x.values()) == 7
    prob += pulp.lpSum(price[i] * x[i] for i in ids) <= budget
    for s in SLOTS:
        prob += pulp.lpSum(x[i] for i in ids if pos[i] == s) >= 1
    for members in by_team.values():
        prob += pulp.lpSum(x[i] for i in members) <= max_team
    for old in prior:
        # Players capped by the exposure rule are not in this pool, so only count the ones that are.
        prob += pulp.lpSum(x[i] for i in old if i in x) <= overlap
    for i in locked:
        if i in x:
            prob += x[i] == 1
    if stack != "none":
        y = pulp.LpVariable.dicts("y", list(by_team), cat="Binary")
        prob += pulp.lpSum(y.values()) == 1
        for t, members in by_team.items():
            prob += pulp.lpSum(x[i] for i in members) >= STACK_MIN * y[t]
            if stack == "bring-back":
                back = by_team.get(opp_of.get(t, ""), [])
                prob += pulp.lpSum(x[i] for i in back) >= y[t]
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    return [i for i in ids if x[i].value() > 0.5]


def build_set(
    pool, budget, mode, n, stack, lam, max_team, overlap, locked, banned, opp_of
):
    """Greedy diverse set: each lineup shares at most `overlap` players with every earlier one,
    and no player appears in more than 60% of lineups (locked players are exempt)."""
    prior, out, counts = [], [], Counter()
    cap = max(1, int(np.ceil(0.6 * n)))
    base = pool[~pool.pid.isin(banned)]
    auto = mode.startswith("Auto")
    for k in range(n):
        st, l = SCHEDULE[k % len(SCHEDULE)] if auto else (stack, lam)
        capped = {i for i, c in counts.items() if c >= cap and i not in locked}
        p = base[~base.pid.isin(capped)]
        ids = solve_one(
            p, _value(p, mode, l), budget, st, max_team, overlap, prior, locked, opp_of
        )
        if ids is None:
            break
        prior.append(ids)
        counts.update(ids)
        out.append((ids, st))
    return out


def summarise(pool, ids, stack):
    rows = pool[pool.pid.isin(ids)].sort_values(["pos", "ev"], ascending=[True, False])
    counts = rows.team.value_counts()
    team = counts.index[0] if counts.iloc[0] >= STACK_MIN else "—"
    return rows, {
        "cost": rows.price.sum(),
        "ev": rows.ev.sum(),
        "own": rows.own.sum(),
        "stack": stack,
        "team": team,
    }


def to_fanteam_csv(pool, out, tournament_id=None):
    """Format solved lineups for FanTeam's Beta CSV uploader.
    Format per line: TournamentID,player1ID,player2ID,player3ID,player4ID,player5ID,player6ID,player7ID
    No header row, no spaces after commas, no trailing comma, and no trailing newline."""
    tid = str(tournament_id).strip() if tournament_id else ""
    if not tid and "tid" in pool.columns and not pool["tid"].empty:
        tid = str(pool["tid"].iloc[0]).strip()
    if not tid:
        raise ValueError("Tournament ID is required for FanTeam CSV upload.")

    slot_rank = {s: idx for idx, s in enumerate(SLOTS)}
    lines = []
    for item in out:
        ids = item[0] if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], (list, set, tuple)) else item
        rows = pool[pool.pid.isin(ids)].copy()
        rows["_slot"] = rows["pos"].map(slot_rank).fillna(99)
        rows = rows.sort_values(["_slot", "ev"], ascending=[True, False])
        ordered_ids = [str(pid).strip() for pid in rows["pid"].tolist()]
        lines.append(f"{tid}," + ",".join(ordered_ids))

    return "\n".join(lines)


def export_fanteam_csv(pool, out, tournament_id=None, filename="fanteam_upload.csv"):
    """Write the FanTeam bulk-upload CSV directly to disk and return the formatted string."""
    csv_text = to_fanteam_csv(pool, out, tournament_id=tournament_id)
    with open(filename, "w", encoding="utf-8", newline="") as f:
        f.write(csv_text)
    return csv_text
