import pandas as pd
import streamlit as st

from optimizer import build_set, load_players, summarise

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
        f'<span><b>{r["ev"]:.1f}</b> pts</span><span><b>{r["own"]:.1f}</b>%</span></div></div>'
        for _, r in rows.iterrows()
    )
    head = (
        f'<div class="head"><div>{tag}<span class="lbl">Lineup {num}</span></div>'
        f'<div class="stats"><span>Cost <b>{s["cost"]:.1f}m</b></span>'
        f'<span>Proj <b>{s["ev"]:.1f}</b></span><span>Own Σ <b>{s["own"]:.0f}%</b></span>'
        f'<span>Stack <b>{s["stack"]} {s["team"]}</b></span></div></div>'
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
    sharp = st.slider("Ownership sharpness", 0.02, 0.15, 0.08, 0.01)

st.markdown('<div class="hero"><h1>Courtside</h1><p>Fanteam NBA · main slate lineup engine</p></div>',
            unsafe_allow_html=True)

c1, c2 = st.columns(2)
dft = c1.file_uploader("Projections · Daily Fantasy Tips CSV", type="csv")
fan = c2.file_uploader("Player list · Fanteam CSV", type="csv")
if not (fan and dft):
    st.info("Upload tonight's Fanteam player list and Daily Fantasy Tips projections to begin.")
    st.stop()

try:
    pool = load_players(fan, dft, sharp)
except Exception as e:  # surface file problems instead of crashing the page
    st.error(f"Could not read the files: {e}")
    st.stop()

opp_of = {t: o for t, o in pool[["team", "opp"]].drop_duplicates().itertuples(index=False)
          if isinstance(o, str)}
m1, m2, m3 = st.columns(3)
m1.metric("Eligible players", len(pool))
m2.metric("Teams", pool.team.nunique())
m3.metric("Budget", f"{budget:.1f}m")

names = sorted(pool["name"])
ex = st.multiselect("Exclude", names)
lk = st.multiselect("Lock in", names)
banned = set(pool.loc[pool.name.isin(ex), "pid"])
locked = set(pool.loc[pool.name.isin(lk), "pid"])

if st.button("Build lineups", type="primary"):
    with st.spinner("Solving lineups…"):
        sets = build_set(pool, budget, mode, n, stack, lam, max_team, overlap, locked, banned, opp_of)
    if not sets:
        st.error("No feasible lineup. Loosen the budget, team cap or locks.")
    export = []
    for num, (ids, kind) in enumerate(sets, 1):
        rows, s = summarise(pool, ids, kind)
        st.markdown(lineup_card(num, rows, s, auto and num <= 3), unsafe_allow_html=True)
        export += [{"lineup": num, "fanteam_id": r.pid, "player": r["name"], "pos": r.pos,
                    "team": r.team, "price": r.price} for _, r in rows.iterrows()]
    if export:
        st.download_button("Download lineups CSV", pd.DataFrame(export).to_csv(index=False),
                           "courtside_lineups.csv", "text/csv")
