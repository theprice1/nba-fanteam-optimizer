from datetime import date

import pandas as pd
import streamlit as st

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
    snapshot,
)

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
if not (fan and dft):
    st.info(
        "Upload tonight's Fanteam player list and Daily Fantasy Tips projections to begin."
    )
    st.stop()

try:
    pool = load_players(fan, dft, sharp)
except (
    KeyError,
    ValueError,
) as e:  # missing columns or unreadable files, shown in the page
    st.error(f"Could not read the files: {e}")
    st.stop()

if "hist" not in st.session_state:
    st.session_state.hist = pd.DataFrame(columns=HIST_COLS)

# ---- Ownership & history -------------------------------------------------------------
with st.expander("Ownership & history", expanded=True):
    st.caption(
        "Save each slate's projections before lock, then attach the field's ownership after it "
        "locks. The history grows every night and improves the ownership model. Cloud storage "
        "resets, so download the history and upload it again next session."
    )
    h_up = st.file_uploader("Load saved history (CSV)", type="csv", key="hist_up")
    if (
        h_up is not None
        and st.session_state.get("hist_src") != f"{h_up.name}-{h_up.size}"
    ):
        st.session_state.hist = load_hist(h_up)
        st.session_state.hist_src = f"{h_up.name}-{h_up.size}"

    slate = st.date_input("Slate date", value=date.today()).isoformat()
    if st.button("Save projections for this slate"):
        st.session_state.hist = snapshot(pool, slate, st.session_state.hist)
        st.success(f"Saved {len(pool)} players for {slate}.")

    paste = st.text_area(
        "Paste the Fanteam ownership table",
        height=150,
        placeholder="Copy the table from Fanteam and paste it here",
    )
    own_csv = st.file_uploader("…or upload an ownership CSV", type="csv", key="own_up")
    saved = sorted(st.session_state.hist.slate.unique())
    target = st.selectbox("Attach to slate", saved) if saved else None
    if st.button("Attach ownership", disabled=target is None):
        if paste.strip():
            field, bad = parse_paste(paste)
        elif own_csv is not None:
            field, bad = parse_csv(own_csv)
        else:
            field, bad = None, 0
        if field is None or field.empty:
            st.warning("Paste the table or upload a CSV first.")
        else:
            st.session_state.hist, matched = attach(
                st.session_state.hist, target, field
            )
            total = field.own.sum()
            st.success(
                f"Attached {len(field)} players to {target}; {matched} matched. "
                f"Ownership sums to {total:.0f}%."
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
    if not hist.empty:
        st.download_button(
            "Download history CSV",
            hist.to_csv(index=False),
            "courtside_history.csv",
            "text/csv",
        )
        rates = play_rates(hist)
        if not rates.empty:
            st.caption(
                "Share of each status that scored (from slates with ownership attached)"
            )
            st.dataframe(rates)

# ---- Build -------------------------------------------------------------------------------
pool = pool.assign(own=calibrated_own(pool, beta) if beta is not None else pool.own)
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

m1, m2, m3 = st.columns(3)
m1.metric("Eligible players", len(pool))
m2.metric("Teams", pool.team.nunique())
m3.metric("Budget", f"{budget:.1f}m")

if st.button("Build lineups", type="primary"):
    with st.spinner("Solving lineups…"):
        sets = build_set(
            pool, budget, mode, n, stack, lam, max_team, overlap, locked, banned, opp_of
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
