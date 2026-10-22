"""Courtside ownership: parse field ownership, keep a daily history, calibrate from it."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from optimizer import norm

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
MIN_SLATES = 3  # calibrate only after several slates, so one unusual night cannot drive the model
MIN_ROWS = 90


NUM_COLS = ["price", "proj", "prior_own", "own", "score"]


def _num(h):
    h = h.copy()
    for c in NUM_COLS:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    return h


def parse_paste(text):
    """Read rows copied from the Fanteam ownership table. Returns (field, lines_not_read)."""
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
                "team": m["team"],
                "pos": m["pos"],
                "own": 0.0 if own == "N/A" else float(own.rstrip("%")),
                "score": float(score) if re.fullmatch(r"[\d.]+", score) else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=["name", "team", "pos", "own", "score"]), bad


def parse_csv(f):
    """CSV with columns Player, Pos, Team, Price, Own %, Score (Form ignored)."""
    df = pd.read_csv(f, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    own = (
        df["own %"]
        .fillna("0")
        .str.replace("%", "", regex=False)
        .str.replace("N/A", "0")
    )
    return pd.DataFrame(
        {
            "name": df["player"].str.strip(),
            "team": df["team"].str.strip().str.upper(),
            "pos": df["pos"].str.strip().str.upper(),
            "own": pd.to_numeric(own, errors="coerce").fillna(0.0),
            "score": pd.to_numeric(df["score"], errors="coerce"),
        }
    ), 0


def load_hist(f):
    h = pd.read_csv(f, encoding="utf-8-sig")
    for c in ["price", "proj", "prior_own", "own", "score"]:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    h["slate"] = h["slate"].astype(str)
    return h[HIST_COLS]


def snapshot(pool, slate, hist):
    """Save this slate's projections, statuses and prices, replacing any earlier save for the slate."""
    rows = pd.DataFrame(
        {
            "slate": slate,
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
    # Keep any ownership and points already attached to this slate, so re-saving projections
    # never wipes the true ownership.
    old = hist[hist.slate == slate] if len(hist) else hist
    if len(old):
        prev = old.drop_duplicates("key").set_index("key")
        rows["own"] = rows["key"].map(prev["own"]).to_numpy()
        rows["score"] = rows["key"].map(prev["score"]).to_numpy()
    kept = hist[hist.slate != slate] if len(hist) else hist
    return _num(pd.concat([kept, rows], ignore_index=True))


def save_hist(hist, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hist[HIST_COLS].to_csv(path, index=False)


def attach(hist, slate, field):
    """Fill ownership and points for a saved slate. Anyone not in the table counts as 0%."""
    if field.empty:
        return hist, 0
    f = (
        field.assign(key=field.name.map(norm) + "|" + field.team)
        .drop_duplicates("key")
        .set_index("key")
    )
    mask = hist.slate == slate
    keys = hist.loc[mask, "key"]
    hist = hist.copy()
    hist.loc[mask, "own"] = keys.map(f["own"]).fillna(0.0).to_numpy()
    hist.loc[mask, "score"] = keys.map(f["score"]).to_numpy()
    return _num(hist), int(keys.isin(f.index).sum())


def _design(df):
    proj = np.asarray(df.proj, dtype=float)
    price = np.asarray(df.price, dtype=float)
    possible = (df.status == "possible").to_numpy(dtype=float)
    return np.column_stack([np.ones(len(df)), proj, price, possible])


def fit(hist):
    """Log-linear ownership model on projection, price and possible status."""
    h = hist.dropna(subset=["own", "proj", "price", "status"])
    if h.slate.nunique() < MIN_SLATES or len(h) < MIN_ROWS:
        return None
    y = np.log(np.asarray(h.own, dtype=float) + 0.5)
    beta, *_ = np.linalg.lstsq(_design(h), y, rcond=None)
    return beta


def calibrated_own(pool, beta):
    raw = np.exp(_design(pool) @ beta)
    return np.minimum(700 * raw / raw.sum(), 100.0)


def report(hist):
    """Projection accuracy over saved slates where the player scored."""
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
    """Share of each lineup status that actually scored, from slates with ownership attached."""
    h = hist[hist.own.notna()].copy()
    if h.empty:
        return pd.DataFrame()
    h["played"] = h.score.fillna(0) > 0
    return (
        h.groupby("status")
        .agg(players=("played", "size"), played_rate=("played", "mean"))
        .round(2)
    )


def slate_summary(hist):
    """One row per saved slate: coverage, projection error, and how far the prior ownership was."""
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
                "prior_own_error": round(
                    float((both.prior_own - both.own).abs().mean()), 2
                )
                if len(both)
                else np.nan,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "slate",
            "players",
            "projections",
            "ownership_attached",
            "mae",
            "bias",
            "prior_own_error",
        ],
    )
