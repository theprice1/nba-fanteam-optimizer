from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from intel import (
    best_per_position,
    error_trend,
    history_scatter,
    leverage_board,
    leverage_chart,
    own_vs_points_chart,
    team_chart,
    team_totals,
    top_owned,
    value_board,
)
from optimizer import build_set, load_players, summarise
from ownership import (
    HIST_COLS,
    attach,
    calibrated_own,
    fit,
    load_hist,
    parse_csv,
    parse_paste,
    play_rates,
    report,
    save_hist,
    slate_summary,
    snapshot,
)

HIST_PATH = Path(__file__).parent / "data" / "courtside_history.csv"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;800&family=Barlow:wght@400;600&display=swap');
:root{--navy:#0b1426;--panel:#121d33;--blue:#1d428a;--orange:#ff7a18;--ink:#eef2fb;--muted:#8a97b5}
.stApp{background:radial-gradient(1100px 520px at 0% -10%,#1d428a66,transparent),linear-gradient(180deg,#070d1b,#0b1426);color:var(--ink);font-family:Barlow,sans-serif}
h1,h2,h3{font-family:'Barlow Condensed',sans-serif;text-transform:uppercase;letter-spacing:.04em}
.hero{background:linear-gradient(120deg,var(--blue),#0b1426 65%);border:1px solid #ffffff14;border-left:6px solid var(--orange);border-radius:18px;padding:20px 26px;margin-bottom:18px}
.hero h1{margin:0;font-size:2.6rem;font-weight:800}
.hero p{margin:2px 0 0;color:var(--muted)}
.card{background:var(--panel);border:1px solid #ffffff14;border-radius:18px;padding:16px 18px;margin-bottom:14px;box-shadow:0 12px 30px #00000066}
.head{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:12px}
.lbl{font-family:'Barlow Condensed';font-weight:800;font-size:1.4rem;text-transform:uppercase}
.tag{font-family:'Barlow Condensed';font-weight:800;color:var(--navy);background:var(--orange);border-radius:999px;padding:2px 12px;margin-right:8px}
.stats{display:flex;gap:16px;color:var(--muted);font-size:.9rem}
.stats b{color:var(--ink);font-family:'Barlow Condensed';font-size:1.15rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px}
.pl{background:var(--navy);border:1px solid #ffffff18;border-radius:14px;padding:10px 12px;border-top:3px solid var(--blue)}
.pos{font-family:'Barlow Condensed';font-weight:800;color:var(--orange);font-size:.85rem;letter-spacing:.06em}
.nm{font-weight:600;margin:2px 0 6px}
.nums{display:flex;justify-content:space-between;font-family:'Barlow Condensed';font-size:1.02rem;color:var(--muted)}
.nums b{color:var(--ink)}
"""


def lineup_card(num, rows, s, main):
    tag = '<span class="tag">MAIN</span>' if main else ""
    chips = "".join(
        f'<div class="pl"><div class="pos">{r["pos"]} · {r["team"]}</div>'
        f'<div class="nm">{r["name"]}</div>'
        f'<div class="nums"><span><b>{r["price"]:.1f}</b>m</span>'
        f"<span><b>{r['ev']:.1f}</b> pts</span><span><b>{r['own']:.1f}</b>%</span></div></div>"
        for _, r in rows.iterrows()
    )
    head = (
        f'<div class="head"><div>{tag}<span class="lbl">Lineup {num}</span></div>'
        f'<div class="stats"><span>Cost <b>{s["cost"]:.1f}m</b></span>'
        f"<span>Proj <b>{s['ev']:.1f}</b></span><span>Own Σ <b>{s['own']:.0f}%</b></span>"
        f"<span>Stack <b>{s['stack']} {s['team']}</b></span></div></div>"
    )
    return f'<div class="card">{head}<div class="grid">{chips}</div></div>'


st.set_page_config(page_title="Courtside · Fanteam NBA", page_icon="🏀", layout="wide")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Contest")
    budget = st.number_input("Budget (m)", 50.0, 120.0, 85.0, 0.5)
    mode = st.radio("Mode", ["Auto · Tournament", "GPP", "Cash", "Pure projection"])
    auto = mode.startswith("Auto")
    n = st.slider("Lineups", 1, 10, 10 if auto else 3)
    stack = "none" if auto else st.selectbox("Stacking", ["team", "bring-back", "none"])
    lam = 0.10 if auto else st.slider("Ownership penalty λ", 0.0, 0.3, 0.10, 0.01)
    max_team = st.slider("Max per team", 2, 5, 4)
    overlap = st.slider("Max shared players", 3, 6, 4)
    sharp = st.slider("Ownership sharpness (prior)", 0.02, 0.15, 0.08, 0.01)

st.markdown(
    '<div class="hero"><h1>Courtside</h1><p>Fanteam NBA · main slate lineup engine</p></div>',
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)
dft = c1.file_uploader("Projections · Daily Fantasy Tips CSV", type="csv")
fan = c2.file_uploader("Player list · Fanteam CSV", type="csv")

pool = None
if fan and dft:
    try:
        pool = load_players(fan, dft, sharp)
    except (KeyError, ValueError) as e:
        st.error(f"Could not read the files: {e}")
else:
    st.info(
        "Upload tonight's player list and projections to unlock the daily intel and the "
        "lineup builder. The ownership section works without them."
    )

if "hist" not in st.session_state:
    st.session_state.hist = (
        load_hist(HIST_PATH) if HIST_PATH.exists() else pd.DataFrame(columns=HIST_COLS)
    )

# ---- Ownership & history: always visible --------------------------------------------
with st.container(border=True):
    st.subheader("Ownership & history")
    h_up = st.file_uploader("Load saved history (CSV)", type="csv", key="hist_up")
    if (
        h_up is not None
        and st.session_state.get("hist_src") != f"{h_up.name}-{h_up.size}"
    ):
        st.session_state.hist = load_hist(h_up)
        st.session_state.hist_src = f"{h_up.name}-{h_up.size}"
        save_hist(st.session_state.hist, HIST_PATH)

    left, right = st.columns(2)
    with left:
        st.markdown("**Save projections for a slate**")
        slate = st.date_input("Slate date", value=date.today()).isoformat()
        if st.button("Save projections", disabled=pool is None):
            st.session_state.hist = snapshot(pool, slate, st.session_state.hist)
            save_hist(st.session_state.hist, HIST_PATH)
            st.success(f"Saved {len(pool)} players for {slate}.")
    with right:
        st.markdown("**Attach ownership to a saved slate**")
        paste = st.text_area(
            "Paste the Fanteam ownership table",
            height=150,
            placeholder="Copy the table from Fanteam and paste it here",
        )
        own_csv = st.file_uploader(
            "…or upload an ownership CSV", type="csv", key="own_up"
        )
        saved = sorted(st.session_state.hist.slate.unique())
        target = st.selectbox("Attach to slate", saved) if saved else None
        if st.button("Attach ownership", disabled=target is None):
            if paste.strip():
                field, bad = parse_paste(paste)
            elif own_csv is not None:
                try:
                    field, bad = parse_csv(own_csv)
                except (KeyError, ValueError) as e:
                    st.error(str(e))
                    field, bad = None, 0
            else:
                field, bad = None, 0
            if field is None or field.empty:
                st.warning("Paste the table or upload a CSV first.")
            else:
                st.session_state.hist, matched = attach(
                    st.session_state.hist, target, field
                )
                save_hist(st.session_state.hist, HIST_PATH)
                total = field.own.sum()
                st.success(
                    f"Attached {len(field)} players to {target}; {matched} matched. "
                    f"Ownership sums to {total:.0f}%."
                )
                if matched < len(field):
                    st.warning(
                        f"{len(field) - matched} ownership rows did not match a player on this slate. "
                        "Check the slate date and the file."
                    )
                if abs(total - 700) > 10:
                    st.warning(
                        "Ownership should sum to about 700%. Check for missing rows."
                    )
                if bad:
                    st.caption(f"{bad} lines could not be read.")

    hist = st.session_state.hist
    beta = fit(hist)
    rep = report(hist)
    m = st.columns(4)
    m[0].metric("Slates saved", hist.slate.nunique())
    m[1].metric("Player-slates", len(hist))
    m[2].metric("Projection error (MAE)", f"{rep['mae']:.1f}" if rep else "—")
    m[3].metric("Ownership model", "Calibrated" if beta is not None else "Prior")
    st.caption(
        "The ownership model calibrates after 3 saved slates with ownership attached."
    )

    with st.expander("Wipe history"):
        st.caption(
            "Deletes every saved slate, projection and ownership figure from this app. Use it only to start again."
        )
        wipe_ok = st.checkbox(
            "I understand this deletes all saved history", key="wipe_ok"
        )
        if wipe_ok and st.button("Wipe history now"):
            HIST_PATH.unlink(missing_ok=True)
            st.session_state.hist = pd.DataFrame(columns=HIST_COLS)
            st.rerun()
    history_tab, detail_tab = st.tabs(["Slate history", "Slate detail"])
    with history_tab:
        summary = slate_summary(hist)
        if summary.empty:
            st.caption(
                "No history yet. Save a slate's projections, then attach its ownership."
            )
        else:
            st.dataframe(summary, hide_index=True)
            if summary.mae.notna().sum() >= 2:
                st.altair_chart(error_trend(summary))
            if rep:
                st.altair_chart(history_scatter(hist))
            rates = play_rates(hist)
            if not rates.empty:
                st.caption(
                    "Share of each status that scored, from slates with ownership attached"
                )
                st.dataframe(rates)
            st.download_button(
                "Download history CSV",
                hist.to_csv(index=False),
                "courtside_history.csv",
                "text/csv",
            )
    with detail_tab:
        slates = sorted(hist.slate.unique())
        if slates:
            view = st.selectbox(
                "Slate", slates, index=len(slates) - 1, key="view_slate"
            )
            st.altair_chart(own_vs_points_chart(hist, view))
            st.dataframe(top_owned(hist, view), hide_index=True)
        else:
            st.caption("Nothing to show yet.")

# ---- Daily intel: only once the player list and projections are loaded -------------
if pool is not None:
    pool = pool.assign(own=calibrated_own(pool, beta) if beta is not None else pool.own)
    st.subheader("Daily intel")
    st.caption(
        f"Projected ownership comes from the {'calibrated model' if beta is not None else 'prior'}. "
        "Expected points allow for injured teammates through the safety nets."
    )
    k = st.columns(4)
    k[0].metric("Eligible players", len(pool))
    k[1].metric("Teams", pool.team.nunique())
    k[2].metric("Top expected points", f"{pool.ev.max():.1f}")
    k[3].metric("Budget", f"{budget:.1f}m")

    v_tab, lev_tab, team_tab, pos_tab = st.tabs(
        ["Value", "Leverage", "Teams", "Positions"]
    )
    with v_tab:
        st.caption(
            "Expected points per million. Cheap players near the top are where the budget works hardest."
        )
        st.dataframe(value_board(pool), hide_index=True)
    with lev_tab:
        st.caption(
            "Top-left is the leverage zone: high expected points, low projected ownership."
        )
        st.altair_chart(leverage_chart(pool))
        st.dataframe(leverage_board(pool), hide_index=True)
    with team_tab:
        st.caption(
            "Teams with the highest projected totals are the usual stacking candidates."
        )
        st.altair_chart(team_chart(team_totals(pool)))
    with pos_tab:
        st.caption("Best three per position by expected points per million.")
        st.dataframe(best_per_position(pool), hide_index=True)

    # ---- Build lineups -----------------------------------------------------------------
    st.subheader("Build lineups")
    names = sorted(pool["name"])
    ex = st.multiselect("Exclude", names)
    lk = st.multiselect("Lock in", names)
    banned = set(pool.loc[pool.name.isin(ex), "pid"])
    locked = set(pool.loc[pool.name.isin(lk), "pid"])
    opp_of = {
        t: o
        for t, o in pool[["team", "opp"]].drop_duplicates().itertuples(index=False)
        if isinstance(o, str)
    }

    if st.button("Build lineups", type="primary"):
        with st.spinner("Solving lineups…"):
            sets = build_set(
                pool,
                budget,
                mode,
                n,
                stack,
                lam,
                max_team,
                overlap,
                locked,
                banned,
                opp_of,
            )
        if not sets:
            st.error("No feasible lineup. Loosen the budget, team cap or locks.")
        export = []
        for num, (ids, kind) in enumerate(sets, 1):
            rows, s = summarise(pool, ids, kind)
            st.markdown(
                lineup_card(num, rows, s, auto and num <= 3), unsafe_allow_html=True
            )
            export += [
                {
                    "lineup": num,
                    "fanteam_id": r.pid,
                    "player": r["name"],
                    "pos": r.pos,
                    "team": r.team,
                    "price": r.price,
                }
                for _, r in rows.iterrows()
            ]
        if export:
            st.download_button(
                "Download lineups CSV",
                pd.DataFrame(export).to_csv(index=False),
                "courtside_lineups.csv",
                "text/csv",
            )
