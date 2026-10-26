from collections import Counter
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
from optimizer import build_set, load_players, summarise, to_fanteam_csv
import optimizer
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
    validate_field,
)

HIST_PATH = Path(__file__).parent / "data" / "courtside_history.csv"
DAILY_DIR = Path(__file__).parent / "data" / "daily"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;800&family=Barlow:wght@400;600&display=swap');
:root{--navy:#0b1426;--panel:#121d33;--blue:#1d428a;--orange:#ff7a18;--green:#10b981;--cyan:#38bdf8;--ink:#eef2fb;--muted:#8a97b5}
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
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(185px,1fr));gap:10px}
.pl{background:var(--navy);border:1px solid #ffffff18;border-radius:14px;padding:10px 12px;border-top:3px solid var(--blue);transition:transform .12s ease,border-color .12s ease}
.pl:hover{transform:translateY(-2px);border-color:#ffffff38}
.pl-chalk{border-top-color:var(--orange)}
.pl-lev{border-top-color:var(--green)}
.pl-val{border-top-color:var(--cyan)}
.pos-row{display:flex;justify-content:space-between;align-items:center}
.pos{font-family:'Barlow Condensed';font-weight:800;color:var(--orange);font-size:.85rem;letter-spacing:.06em}
.pill{font-family:'Barlow Condensed';font-weight:800;font-size:.72rem;letter-spacing:.05em;padding:1px 7px;border-radius:999px;text-transform:uppercase}
.pill-chalk{background:#ff7a1826;color:var(--orange);border:1px solid #ff7a1866}
.pill-lev{background:#10b98126;color:var(--green);border:1px solid #10b98166}
.pill-val{background:#38bdf826;color:var(--cyan);border:1px solid #38bdf866}
.nm{font-weight:600;margin:3px 0 6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.nums{display:flex;justify-content:space-between;font-family:'Barlow Condensed';font-size:1.02rem;color:var(--muted)}
.nums b{color:var(--ink)}

/* --- High Contrast Mobile & Dark Theme Fixes --- */
label, .stMarkdown p, .stCaption, [data-testid="stMetricLabel"], [data-testid="stText"] {
    color: var(--ink) !important;
}
[data-testid="stMetricValue"] {
    color: var(--orange) !important;
    font-family: 'Barlow Condensed', sans-serif !important;
}
[data-testid="stVerticalBlock"] > [data-testid="stContainer"] {
    background-color: var(--panel);
    border: 1px solid #ffffff14;
    border-radius: 18px;
    padding: 16px;
}
[data-testid="stFileUploader"] {
    background-color: var(--navy);
    border: 1px solid #ffffff22;
    border-radius: 12px;
    padding: 12px;
}
[data-testid="stFileUploader"] section {
    background-color: var(--panel) !important;
}
.stButton button[kind="primary"], button[kind="primary"] {
    background-color: var(--orange) !important;
    color: var(--navy) !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 800 !important;
    border-radius: 12px !important;
}

@media (max-width: 768px) {
    .hero { padding: 14px 18px; }
    .hero h1 { font-size: 2rem; }
    .card { padding: 12px 14px; }
    .grid { grid-template-columns: 1fr; }
    .stats { flex-direction: column; gap: 4px; }
}
"""


def sync_daily_folders(hist, sharp):
    """Lightweight check: only sync daily folders if they aren't already in history."""
    if not DAILY_DIR.exists():
        return hist

    saved_slates = set(hist.slate.unique()) if not hist.empty else set()
    changed = False

    for folder in sorted(DAILY_DIR.iterdir()):
        if not folder.is_dir():
            continue
        slate = folder.name

        if slate in saved_slates:
            slate_rows = hist[hist.slate == slate]
            if not slate_rows.empty and not slate_rows.own.isna().all():
                continue

        fan_f = folder / "fanteam_list.csv"
        dft_f = folder / "projections.csv"
        own_f = folder / "ownership.csv"

        if slate not in saved_slates and fan_f.exists() and dft_f.exists():
            try:
                with fan_f.open("rb") as f1, dft_f.open("rb") as f2:
                    day_pool = load_players(f1, f2, sharp)
                hist = snapshot(day_pool, slate, hist)
                saved_slates.add(slate)
                changed = True
            except (KeyError, ValueError, FileNotFoundError):
                continue

        slate_rows = hist[hist.slate == slate]
        if not slate_rows.empty and slate_rows.own.isna().all() and own_f.exists():
            try:
                with own_f.open("rb") as f3:
                    field, _ = parse_csv(f3)
                if not validate_field(field):
                    hist, _ = attach(hist, slate, field)
                    changed = True
            except (KeyError, ValueError, FileNotFoundError):
                continue

    if changed:
        try:
            save_hist(hist, HIST_PATH)
        except (OSError, PermissionError):
            pass

    return hist


def _player_badge(r):
    val = r["ev"] / max(r["price"], 4.0)
    if r["own"] >= 28.0:
        return "pl-chalk", '<span class="pill pill-chalk">CHALK</span>'
    if r["ev"] >= 32.0 and r["own"] <= 14.0:
        return "pl-lev", '<span class="pill pill-lev">LEVERAGE</span>'
    if val >= 3.45 and r["price"] <= 9.5:
        return "pl-val", '<span class="pill pill-val">VALUE</span>'
    return "", ""


def lineup_card(num, rows, s, main):
    tag = '<span class="tag">MAIN</span>' if main else ""
    chips = []
    for _, r in rows.iterrows():
        cls, badge = _player_badge(r)
        chips.append(
            f'<div class="pl {cls}">'
            f'<div class="pos-row"><span class="pos">{r["pos"]} · {r["team"]}</span>{badge}</div>'
            f'<div class="nm" title="{r["name"]}">{r["name"]}</div>'
            f'<div class="nums"><span><b>{r["price"]:.1f}</b>m</span>'
            f'<span><b>{r["ev"]:.1f}</b> pts</span><span><b>{r["own"]:.1f}</b>%</span></div></div>'
        )
    chips_html = "".join(chips)
    head = (
        f'<div class="head"><div>{tag}<span class="lbl">Lineup {num}</span></div>'
        f'<div class="stats"><span>Cost <b>{s["cost"]:.1f}m</b></span>'
        f"<span>Proj <b>{s['ev']:.1f}</b></span><span>Own Σ <b>{s['own']:.0f}%</b></span>"
        f"<span>Stack <b>{s['stack']} {s['team']}</b></span></div></div>"
    )
    return f'<div class="card">{head}<div class="grid">{chips_html}</div></div>'


def exposure_summary(pool, sets):
    """Build a portfolio exposure and field-leverage table across all generated lineups."""
    total_lineups = max(len(sets), 1)
    counts = Counter()
    for ids, _ in sets:
        counts.update(ids)
    used = pool[pool["pid"].isin(counts)].copy()
    used["Lineups"] = used["pid"].map(counts)
    used["My Exp %"] = (100.0 * used["Lineups"] / total_lineups).round(1)
    used["Field Own %"] = used["own"].round(1)
    used["Leverage %"] = (used["My Exp %"] - used["Field Own %"]).round(1)
    used["Value (Pts/M)"] = (used["ev"] / used["price"]).round(2)
    cols = {
        "name": "Player",
        "pos": "Pos",
        "team": "Team",
        "price": "Price (M)",
        "ev": "Proj EV",
        "Value (Pts/M)": "Value",
        "Lineups": "Lineups",
        "My Exp %": "My Exp %",
        "Field Own %": "Field Own %",
        "Leverage %": "Leverage %",
    }
    out = (
        used[list(cols.keys())]
        .rename(columns=cols)
        .sort_values(["Lineups", "Proj EV"], ascending=[False, False])
    )
    out["Price (M)"] = out["Price (M)"].round(1)
    out["Proj EV"] = out["Proj EV"].round(1)
    return out


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
    overlap = st.slider("Max shared players", 3, 6, 3)
    sharp = st.slider("Ownership sharpness (prior)", 0.02, 0.15, 0.08, 0.01)
    st.sidebar.caption(f"Optimizer file: {optimizer.__file__}")
    st.sidebar.caption(f"Rules: overlap≤{optimizer.MAX_OVERLAP}, exposure≤{optimizer.MAX_SHARE:.0%}, chalk≤{optimizer.MAX_CHALK}")

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
        slate_sig = f"{fan.name}-{fan.size}-{dft.name}-{dft.size}"
        if st.session_state.get("slate_sig") != slate_sig:
            st.session_state.slate_sig = slate_sig
            st.session_state.pop("sets", None)
    except (KeyError, ValueError) as e:
        st.error(f"Could not read the files: {e}")
else:
    st.info(
        "Upload tonight's player list and projections to unlock the daily intel and the "
        "lineup builder. The ownership section works without them."
    )

if "hist" not in st.session_state:
    base_hist = (
        load_hist(HIST_PATH) if HIST_PATH.exists() else pd.DataFrame(columns=HIST_COLS)
    )
    st.session_state.hist = sync_daily_folders(base_hist, sharp)

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
            problems = (
                validate_field(field) if field is not None and not field.empty else []
            )
            if field is None or field.empty:
                st.warning("Paste the table or upload a CSV first.")
            elif problems:
                for problem in problems:
                    st.error(problem)
                st.caption("Nothing was attached.")
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
    beta = None  # prior is the default: walk-forward it matched or beat the calibrated model (week 1)
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
        f"Projected ownership comes from the {'calibrated model (with Form & Value scaling)' if beta is not None else 'multi-factor prior'}. "
        "Expected points allow for injured starters through price-capped safety nets."
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
    default_tid = (
        str(pool["tid"].iloc[0]).strip()
        if "tid" in pool.columns and not pool["tid"].empty
        else ""
    )
    tid = st.text_input(
        "Tournament ID (auto-filled from Fanteam CSV; change if entering another contest on the same slate)",
        value=default_tid,
    ).strip()

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
        if sets and len(sets) < n:
            st.warning(f"Only {len(sets)} of {n} lineups could be built under the rules. Try a higher budget or fewer locks.")
        if optimizer.RELAXED:
            st.info("Relaxed to build: " + ", ".join(f"lineup {k} ({why})" for k, why in optimizer.RELAXED))
        if not sets:
            st.error("No feasible lineup. Loosen the budget, team cap or locks.")
            st.session_state.pop("sets", None)
        else:
            st.session_state.sets = sets

    active_sets = st.session_state.get("sets")
    if active_sets:
        dl1, dl2 = st.columns(2)
        try:
            fanteam_csv = to_fanteam_csv(pool, active_sets, tournament_id=tid)
            dl1.download_button(
                f"Download FanTeam Upload CSV ({tid or 'Set ID'})",
                fanteam_csv,
                f"fanteam_upload_{tid or 'lineups'}.csv",
                "text/csv",
                type="primary",
                use_container_width=True,
            )
        except ValueError as e:
            dl1.warning(str(e))

        export_rows = []
        for num, (ids, kind) in enumerate(active_sets, 1):
            rows, _ = summarise(pool, ids, kind)
            for _, r in rows.iterrows():
                export_rows.append(
                    {
                        "lineup": num,
                        "fanteam_id": r.pid,
                        "player": r["name"],
                        "pos": r.pos,
                        "team": r.team,
                        "price": r.price,
                        "ev": round(float(r.ev), 1),
                        "own": round(float(r.own), 1),
                    }
                )
        dl2.download_button(
            "Download Readable Audit Table (CSV)",
            pd.DataFrame(export_rows).to_csv(index=False),
            "courtside_lineups_audit.csv",
            "text/csv",
            use_container_width=True,
        )

        with st.expander("Portfolio exposure & field leverage summary", expanded=True):
            st.caption(
                "Positive Leverage % indicates you are overweight relative to projected field ownership; "
                "negative Leverage % indicates you are underweight on field chalk."
            )
            st.dataframe(exposure_summary(pool, active_sets), hide_index=True, use_container_width=True)

        for num, (ids, kind) in enumerate(active_sets, 1):
            rows, s = summarise(pool, ids, kind)
            st.markdown(
                lineup_card(num, rows, s, auto and num <= 3), unsafe_allow_html=True
            )
