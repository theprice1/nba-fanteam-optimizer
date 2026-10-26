"""Courtside ownership: parse field ownership, keep a daily history, calibrate from it."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from optimizer import AVAIL, norm, team_code

ROW = re.compile(
    r"^(?P<name>.+?)\s+(?P<pos>PG|SG|SF|PF|C)\s+(?P<team>[A-Z]{2,3})\s+"
    r"(?P<price>[\d.]+)M?\s+(?P<own>[\d.]+%|N/A)\s+(?P<form>\S+)\s+(?P<score>\S+)$"
)
HIST_COLS = [
    "slate",
    "key",
    "name",
    "team",
    "pos",
    "price",
    "proj",
    "status",
    "prior_own",
    "own",
    "score",
]
MIN_SLATES = 3
MIN_ROWS = 90

NUM_COLS = ["price", "proj", "prior_own", "own", "score"]


def _num(h):
    h = h.copy()
    for c in NUM_COLS:
        if c in h.columns:
            h[c] = pd.to_numeric(h[c], errors="coerce")
    return h


def _find(df, names, label):
    keys = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}
    for n in names:
        if n in keys:
            return keys[n]
    raise ValueError(f"The ownership file needs a {label} column.")


ZERO_TOKENS = {"-", "–", "—"}
POINTS = re.compile(r"[+-]?\d+(\.\d+)?")


def _points(s):
    """Fanteam shows zero-point players with a dash (or 0.00). Both read as zero;
    anything else that is not a number stays missing."""
    s = str(s).strip()
    if s in ZERO_TOKENS:
        return 0.0
    return float(s) if POINTS.fullmatch(s) else np.nan


def parse_paste(text):
    rows, bad = [], 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("player"):
            continue
        m = ROW.match(line)
        if not m:
            bad += 1
            continue
        own, score = m["own"], m["score"]
        rows.append(
            {
                "name": m["name"].strip(),
                "team": team_code(m["team"]),
                "pos": m["pos"],
                "own": 0.0 if own == "N/A" else float(own.rstrip("%")),
                "score": _points(score),
            }
        )
    return pd.DataFrame(rows, columns=["name", "team", "pos", "own", "score"]), bad


def parse_csv(f):
    if hasattr(f, "seek"):
        f.seek(0)
    df = pd.read_csv(f, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if {c.lower() for c in df.columns} & {"slate", "key", "prior_own"}:
        raise ValueError("This looks like a saved history file.")
    name_c = _find(df, ["player", "name"], "Player")
    team_c = _find(df, ["team", "club"], "Team")
    own_c = _find(df, ["own", "ownpct", "ownership"], "Own %")
    score_c = _find(df, ["score", "points"], "Score")
    own = (
        df[own_c]
        .fillna("0")
        .str.replace("%", "", regex=False)
        .str.replace("N/A", "0", regex=False)
    )
    out = pd.DataFrame(
        {
            "name": df[name_c].str.strip(),
            "team": df[team_c].map(team_code),
            "own": pd.to_numeric(own, errors="coerce").fillna(0.0),
            "score": df[score_c].map(_points).astype(float),
        }
    )
    # Remove exact key duplicates by keeping the row with highest ownership / score
    out["_key"] = out["name"].map(norm) + "|" + out["team"]
    out = (
        out.sort_values(["own", "score"], ascending=[False, False])
        .drop_duplicates("_key", keep="first")
        .drop(columns=["_key"])
        .reset_index(drop=True)
    )
    return out, 0


def validate_field(field):
    problems = []
    total = float(field.own.sum())
    if not 650 <= total <= 750:
        problems.append(
            f"Ownership adds up to {total:.0f}%, but it should be about 700%. "
            "The file may be the wrong one or incomplete."
        )
    key = field.name.map(norm) + "|" + field.team
    if key.duplicated().any():
        problems.append(
            "The file lists some players twice. Check it is the ownership table for one slate."
        )
    return problems


def load_hist(f):
    if hasattr(f, "seek"):
        f.seek(0)
    h = pd.read_csv(f, encoding="utf-8-sig")
    for c in NUM_COLS:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    h["slate"] = h["slate"].astype(str)
    return h[HIST_COLS]


def snapshot(pool, slate, hist):
    rows = pd.DataFrame(
        {
            "slate": str(slate),
            "key": (pool.key + "|" + pool.team).to_numpy(),
            "name": pool.name.to_numpy(),
            "team": pool.team.to_numpy(),
            "pos": pool.pos.to_numpy(),
            "price": pool.price.to_numpy(),
            "proj": pool.proj.to_numpy(),
            "status": pool.status.to_numpy(),
            "prior_own": pool.own.to_numpy(),
            "own": np.nan,
            "score": np.nan,
        }
    )
    old = hist[hist.slate == str(slate)] if len(hist) else hist
    if len(old):
        prev = old.drop_duplicates("key").set_index("key")
        rows["own"] = rows["key"].map(prev["own"]).to_numpy()
        rows["score"] = rows["key"].map(prev["score"]).to_numpy()
    kept = hist[hist.slate != str(slate)] if len(hist) else hist
    return _num(pd.concat([kept, rows], ignore_index=True)[HIST_COLS])


def save_hist(hist, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hist[HIST_COLS].to_csv(path, index=False)


def attach(hist, slate, field):
    if field.empty:
        return hist, 0
    f = (
        field.assign(key=field.name.map(norm) + "|" + field.team.map(team_code))
        .drop_duplicates("key")
        .set_index("key")
    )
    mask = hist.slate == str(slate)
    keys = hist.loc[mask, "key"]
    hist = hist.copy()
    hist.loc[mask, "own"] = keys.map(f["own"]).fillna(0.0).to_numpy()
    hist.loc[mask, "score"] = keys.map(f["score"]).to_numpy()
    return _num(hist), int(keys.isin(f.index).sum())


def _design(df):
    """Within-slate standardized design matrix capturing projection ceiling,
    salary efficiency (value = proj/price), positional top-2 scarcity, and starter status."""
    proj = np.asarray(df.proj, dtype=float)
    price = np.maximum(np.asarray(df.price, dtype=float), 4.0)
    val = proj / price

    z_proj = (proj - proj.mean()) / (proj.std() + 1e-6)
    z_price = (price - price.mean()) / (price.std() + 1e-6)
    z_val = (val - val.mean()) / (val.std() + 1e-6)

    pos_s = df["pos"] if "pos" in df.columns else pd.Series("PG", index=df.index)
    val_rank = (
        pd.Series(val, index=df.index)
        .groupby(pos_s)
        .rank(ascending=False)
        .to_numpy()
    )
    proj_rank = (
        pd.Series(proj, index=df.index)
        .groupby(pos_s)
        .rank(ascending=False)
        .to_numpy()
    )
    top_pos_chalk = ((val_rank <= 2) | (proj_rank <= 2)).astype(float)
    not_starter = (~df.status.str.lower().isin(["confirmed", "expected"])).to_numpy(
        dtype=float
    )

    return np.column_stack(
        [np.ones(len(df)), z_proj, z_price, z_val, top_pos_chalk, not_starter]
    )


def fit(hist):
    """Fit slate-size-normalized weighted least squares across saved slates."""
    h = hist.dropna(subset=["own", "proj", "price", "status"]).copy()
    if h.slate.nunique() < MIN_SLATES or len(h) < MIN_ROWS:
        return None

    X_list, y_list, w_list = [], [], []
    for _, grp in h.groupby("slate"):
        if len(grp) < 10:
            continue
        mean_own = 700.0 / len(grp)
        own_arr = np.asarray(grp.own, dtype=float)
        X_list.append(_design(grp))
        # Normalize target by slate size so 41-player and 241-player slates align cleanly
        y_list.append(np.log((own_arr + 0.25) / mean_own))
        # Weight high-owned players higher so chalk calibration dominates
        w_list.append(np.sqrt(own_arr + 1.0))

    if not X_list:
        return None

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    w = np.concatenate(w_list)
    beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    return beta


def calibrated_own(pool, beta):
    """Predict ownership using calibrated historical weights plus live FanTeam Form divergence."""
    log_pred = _design(pool) @ beta

    # Incorporate live FanTeam UI Form signal when available on pool
    if "form" in pool.columns and pool["form"].notna().any():
        proj = np.asarray(pool.proj, dtype=float)
        form = np.asarray(pool.form.fillna(pool.proj), dtype=float)
        z_proj = (proj - proj.mean()) / (proj.std() + 1e-6)
        z_form = (form - form.mean()) / (form.std() + 1e-6)
        # Boost players whose recent Form exceeds their projection (and discount low-Form stars)
        log_pred = log_pred + 0.22 * (z_form - z_proj)

    p_avail = (
        pool["p"].to_numpy(dtype=float)
        if "p" in pool.columns
        else pool["status"].str.lower().map(AVAIL).fillna(0.5).to_numpy(dtype=float)
    )
    raw = np.exp(np.clip(log_pred, -6.0, 6.0)) * p_avail
    return np.minimum(700.0 * raw / raw.sum(), 100.0)


def report(hist):
    h = hist.dropna(subset=["proj", "score"])
    h = h[h.score > 0]
    if h.empty:
        return None
    e = h.score - h.proj
    return {
        "rows": len(h),
        "mae": float(e.abs().mean()),
        "bias": float(e.mean()),
        "rmse": float(np.sqrt((e**2).mean())),
    }


def play_rates(hist):
    """Share of each status that scored. Only matched players with a projection above
    zero count: unmatched rows are not DNPs, and zero-projection players are not expected to play."""
    h = hist[hist.own.notna() & hist.score.notna() & (hist.proj > 0)].copy()
    if h.empty:
        return pd.DataFrame()
    h["played"] = h.score > 0
    return (
        h.groupby("status")
        .agg(players=("played", "size"), played_rate=("played", "mean"))
        .round(2)
    )


def slate_summary(hist):
    rows = []
    for slate, h in hist.groupby("slate"):
        scored = h.dropna(subset=["proj", "score"])
        scored = scored[scored.score > 0]
        e = scored.score - scored.proj
        both = h.dropna(subset=["prior_own", "own"])
        rows.append(
            {
                "slate": slate,
                "players": len(h),
                "projections": int(h.proj.notna().sum()),
                "ownership_attached": int(h.own.notna().sum()),
                "mae": round(float(e.abs().mean()), 2) if len(e) else np.nan,
                "bias": round(float(e.mean()), 2) if len(e) else np.nan,
                "prior_own_error": (
                    round(float((both.prior_own - both.own).abs().mean()), 2)
                    if len(both)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)
