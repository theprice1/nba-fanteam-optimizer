"""Courtside intel: what the projections say about value, leverage and team totals,
plus history charts. Pure pandas and Altair, so it runs with the existing requirements."""

import altair as alt
import pandas as pd

POSITIONS = ["PG", "SG", "SF", "PF", "C"]
STATUS_SCALE = alt.Scale(
    domain=["expected", "confirmed", "possible", "bench", "unexpected"],
    range=["#ff7a18", "#1d8cff", "#8a97b5", "#c084fc", "#ef4444"],
)
PLAYER_COLS = ["name", "team", "pos", "price", "ev", "own", "status"]


def _with_value(pool):
    return pool.assign(value=pool.ev / pool.price)


def value_board(pool, n=15):
    """Best expected points per million spent."""
    return _with_value(pool).nlargest(n, "value")[PLAYER_COLS + ["value"]].round(2)


def leverage_board(pool, n=12, max_own=5.0):
    """High expected points from players the field is barely using."""
    return pool[pool.own <= max_own].nlargest(n, "ev")[PLAYER_COLS].round(1)


def team_totals(pool):
    """Sum of expected points per team, from the eligible players in the pool."""
    return (
        pool.groupby("team")
        .agg(players=("ev", "size"), proj_total=("ev", "sum"), best=("ev", "max"))
        .round(1)
        .sort_values("proj_total", ascending=False)
        .reset_index()
    )


def best_per_position(pool, n=3):
    d = _with_value(pool)
    parts = [d[d.pos == s].nlargest(n, "value").assign(slot=s) for s in POSITIONS]
    return pd.concat(parts)[
        ["slot", "name", "team", "price", "ev", "own", "value"]
    ].round(2)


def leverage_chart(pool):
    return (
        alt.Chart(pool[PLAYER_COLS])
        .mark_circle(size=110, opacity=0.85)
        .encode(
            x=alt.X("own:Q", title="Projected ownership %"),
            y=alt.Y("ev:Q", title="Expected points"),
            color=alt.Color("status:N", scale=STATUS_SCALE, title="Status"),
            tooltip=["name", "team", "pos", "price", "ev", "own", "status"],
        )
        .properties(height=380)
    )


def team_chart(totals):
    return (
        alt.Chart(totals)
        .mark_bar(color="#ff7a18", cornerRadiusEnd=4)
        .encode(
            x=alt.X("proj_total:Q", title="Projected points, eligible players"),
            y=alt.Y("team:N", sort="-x", title=None),
            tooltip=["team", "players", "proj_total", "best"],
        )
        .properties(height=max(200, 22 * len(totals)))
    )


def top_owned(hist, slate, n=15):
    h = hist[(hist.slate == slate) & hist.own.notna()]
    return h.nlargest(n, "own")[
        ["name", "team", "pos", "price", "proj", "own", "score"]
    ].round(1)


def own_vs_points_chart(hist, slate):
    h = hist[(hist.slate == slate) & hist.own.notna() & hist.score.notna()]
    return (
        alt.Chart(h[["name", "own", "score", "proj"]])
        .mark_circle(size=90, opacity=0.8, color="#1d8cff")
        .encode(
            x=alt.X("own:Q", title="True ownership %"),
            y=alt.Y("score:Q", title="Points scored"),
            tooltip=["name", "own", "proj", "score"],
        )
        .properties(height=340)
    )


def history_scatter(hist):
    """Projected vs scored for every saved player with a score, with a y = x reference line."""
    h = hist.dropna(subset=["proj", "score"])
    h = h[h.score > 0][["name", "proj", "score", "slate"]]
    points = (
        alt.Chart(h)
        .mark_circle(size=70, opacity=0.7)
        .encode(
            x=alt.X("proj:Q", title="Projected"),
            y=alt.Y("score:Q", title="Scored"),
            color=alt.Color("slate:N", title="Slate"),
            tooltip=["name", "proj", "score", "slate"],
        )
    )
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0, 60], "y": [0, 60]}))
        .mark_line(color="#8a97b5", strokeDash=[4, 4])
        .encode(x="x:Q", y="y:Q")
    )
    return (points + diagonal).properties(height=340)


def error_trend(summary):
    return (
        alt.Chart(summary.dropna(subset=["mae"]))
        .mark_line(point=True, color="#1d8cff")
        .encode(
            x=alt.X("slate:N", title="Slate"),
            y=alt.Y("mae:Q", title="Projection error (MAE)"),
            tooltip=["slate", "mae", "bias"],
        )
        .properties(height=260)
    )
