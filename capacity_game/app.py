"""Osteogenic-capacity comparison game -- Streamlit UI.

Run from the repo root:
    streamlit run capacity_game/app.py

Two synthetic patient cases are shown side by side; the surgeon picks the one with **higher
osteogenic capacity** (the patient's intrinsic bone-forming ability). The winner stays, the loser
is replaced by a coverage-aware challenger, and every click updates a shared Elo ladder that is
binned into capacity buckets for model training.
"""

from __future__ import annotations

import base64
import contextlib
import hmac
import html
import mimetypes
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import analysis
import case_display
import engine
import exports
import roster
import storage

# --- deployment secrets bridge -------------------------------------------------------------
# Streamlit Community Cloud has no persistent disk, so votes + roster are kept in a SQL database
# when `db_url` is set (in the app's Secrets box or .streamlit/secrets.toml). Bridge that secret
# into the environment the pure-Python engine/roster/storage read, and tag this app's rows with a
# table prefix so the capacity and demand games can safely share one database. With no db_url set
# (local runs / a persistent-disk server) nothing changes -- the app uses local CSV as before.
os.environ.setdefault("BONEGRAFT_TABLE_PREFIX", "capacity_")
with contextlib.suppress(Exception):  # accessing st.secrets raises when no secrets file exists
    _db_url = st.secrets.get("db_url")
    if _db_url and not os.environ.get("BONEGRAFT_DB_URL"):
        os.environ["BONEGRAFT_DB_URL"] = str(_db_url)

# --------------------------------------------------------------------------------- paths --
ROOT = Path(__file__).resolve().parents[1]
# Input cases ship read-only with the app; override the location for deployment if needed.
CASES_PATH = Path(os.environ.get("CAPACITY_CASES_PATH")
                  or (ROOT / "outputs" / "synthetic" / "synthetic_cases.csv"))
# Collected data (votes, roster, case pool, model-ready exports). Point CAPACITY_DATA_DIR at
# PERSISTENT storage when deployed (e.g. /home/data/capacity_game on Azure App Service) so the
# votes and roster survive restarts and redeploys.
OUT_DIR = Path(os.environ.get("CAPACITY_DATA_DIR")
               or (ROOT / "outputs" / "capacity_game"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
VOTES_PATH = engine.votes_path(OUT_DIR)
ROSTER_PATH = roster.roster_path(OUT_DIR)
FLAGS_PATH = engine.flags_path(OUT_DIR)
ASSETS = Path(__file__).resolve().parent / "assets"
_LOGO_FILES = ("medtronic_logo.svg", "medtronic_logo.png", "medtronic-logo.png",
               "medtronic.svg", "medtronic.png", "logo.svg", "logo.png")


@st.cache_data(show_spinner=False)
def _logo_uri() -> str:
    """Base64 data-URI for the official Medtronic logo if the asset is present in
    capacity_game/assets/, else "" (the chrome then falls back to a text wordmark)."""
    for name in _LOGO_FILES:
        p = ASSETS / name
        if p.exists():
            mime = ("image/svg+xml" if p.suffix.lower() == ".svg"
                    else mimetypes.guess_type(str(p))[0] or "image/png")
            return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode('ascii')}"
    return ""


def _brand_lockup(dark: bool, cls: str = "brand-logo") -> str:
    """Official logo on a white chip (so the colour mark reads on any background); text fallback."""
    if LOGO_URI:
        chip = "logo-chip" if dark else "logo-chip-light"
        return f"<span class='{chip}'><img src='{LOGO_URI}' class='{cls}' alt='Medtronic'/></span>"
    tone = "#fff" if dark else "var(--mdt-navy)"
    return f"<span class='brand-name' style='color:{tone}'>Medtronic</span>"


def _admin_passcode() -> str | None:
    """Owner passcode from .streamlit/secrets.toml (admin_passcode) or the
    CAPACITY_ADMIN_PASSCODE env var. Returns None when nothing is configured, which
    keeps owner mode (insights + exports) locked -- fail closed."""
    pc = None
    try:
        pc = st.secrets.get("admin_passcode")  # accessing st.secrets raises if no file exists
    except Exception:
        pc = None
    return pc or os.environ.get("CAPACITY_ADMIN_PASSCODE") or None


def _passcode_ok(entered: str) -> bool:
    pc = _admin_passcode()
    return bool(pc) and hmac.compare_digest(str(entered), str(pc))


STREAK_RETIRE = 6      # after N straight case-wins, retire the champion and draw a fresh pair
POOL_SIZE = 200        # reduced, fixed comparison pool so full coverage is realistic
POOL_SEED = 42
SESSION_MINUTES = 5    # gentle "great batch" nudge after ~this long (configurable in owner mode)
MILESTONES = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500]  # celebrate every N lifetime comparisons

st.set_page_config(
    page_title="Medtronic · Osteogenic Capacity Study",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

SESSION_MINUTES_DEFAULT = SESSION_MINUTES
LOGO_URI = _logo_uri()

# --------------------------------------------------------------------------------- style --
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --mdt-navy:#00174F; --mdt-royal:#1B0FC4; --mdt-blue:#0068B3; --mdt-blue-600:#005A9C;
  --mdt-blue-700:#003E73; --mdt-cyan:#00A9E0; --mdt-magenta:#E6007E; --mdt-bright:#0093D0;
  --mdt-sky:#E7F1FA; --ink:#16233A; --muted:#5B6577;
  --line:#E3E9F2; --bg:#EDF2F8; --card:#FFFFFF;
  --good:#0E7C4A; --good-bg:#E4F4EC; --warn:#B26A00; --warn-bg:#FAF0DE;
  --bad:#C0392B; --bad-bg:#FBEAE8;
}
html, body, .stApp, [class*="css"]{ font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; }
[data-testid="stAppViewContainer"]{ background:var(--bg); }
[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer{ visibility:hidden; }
.block-container{ padding-top:0.6rem; padding-bottom:2rem; max-width:1180px; }

.app-header{ display:flex; align-items:center; justify-content:space-between;
  background:
    radial-gradient(120% 160% at 88% -30%, rgba(230,0,126,.50), transparent 45%),
    radial-gradient(120% 150% at 60% 140%, rgba(0,169,224,.55), transparent 52%),
    linear-gradient(105deg,#14008C 0%,#1B0FC4 55%,#123FCB 100%);
  color:#fff; padding:15px 22px; border-radius:16px; margin-bottom:16px;
  box-shadow:0 10px 26px rgba(20,0,140,.28); }
.app-header .brand{ display:flex; align-items:center; gap:11px; }
.brand-name{ font-weight:800; font-size:19px; letter-spacing:.2px; }
.brand-sep{ opacity:.45; font-weight:300; }
.app-title{ font-weight:500; font-size:14.5px; opacity:.9; }
.header-right{ display:flex; gap:26px; align-items:center; }
.hstat{ text-align:right; line-height:1.1; }
.hstat .v{ font-weight:800; font-size:18px; }
.hstat .l{ font-size:10.5px; opacity:.78; text-transform:uppercase; letter-spacing:.08em; margin-top:2px; }

.sec-title{ font-weight:800; color:var(--mdt-navy); font-size:17px; margin:10px 0 8px;
  position:relative; padding-left:12px; }
.sec-title::before{ content:''; position:absolute; left:0; top:3px; bottom:3px; width:4px;
  border-radius:2px; background:linear-gradient(180deg,var(--mdt-cyan),var(--mdt-magenta)); }

.logo-chip{ background:#fff; border-radius:10px; padding:6px 12px; display:inline-flex;
  align-items:center; box-shadow:0 3px 10px rgba(0,0,0,.20); }
.logo-chip-light{ display:inline-flex; align-items:center; }
.brand-logo{ height:40px; display:block; }
.hero-logo{ height:56px; display:block; }
.side-logo{ height:34px; display:block; }
.hero-badge{ background:#fff; border-radius:12px; padding:9px 15px; display:inline-flex;
  align-items:center; margin-bottom:20px; box-shadow:0 8px 22px rgba(0,0,0,.20); }

.hero{ background:
    radial-gradient(110% 140% at 92% -15%, rgba(230,0,126,.55), transparent 46%),
    radial-gradient(100% 130% at 12% 125%, rgba(0,169,224,.55), transparent 52%),
    linear-gradient(120deg,#12007F,#1B0FC4 55%,#0E3FD0);
  border-radius:22px; padding:40px 44px; color:#fff; box-shadow:0 16px 46px rgba(18,0,127,.30); }
.hero h1{ font-size:32px; font-weight:800; margin:0 0 8px; letter-spacing:-.3px; }
.hero p{ font-size:15.5px; opacity:.93; max-width:660px; line-height:1.55; margin:0; }
.hero-steps{ display:flex; gap:14px; margin-top:24px; flex-wrap:wrap; }
.hero-step{ background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.18);
  border-radius:14px; padding:14px 16px; flex:1; min-width:170px; }
.hero-step .n{ font-weight:800; font-size:11px; opacity:.65; letter-spacing:.1em; }
.hero-step .t{ font-weight:700; margin-top:3px; font-size:15px; }
.hero-step .d{ font-size:12.5px; opacity:.85; margin-top:3px; line-height:1.4; }

.stButton > button{ border-radius:12px; font-weight:600; padding:.55rem 1rem; transition:all .15s ease; }
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"]{
  background:linear-gradient(135deg,#0A63D6,#1B12C4); color:#fff; border:none;
  min-height:54px; font-size:1.02rem; box-shadow:0 6px 18px rgba(27,15,196,.30); }
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover{
  transform:translateY(-1px); box-shadow:0 9px 24px rgba(27,15,196,.40); filter:saturate(1.06) brightness(1.03); }
.stButton > button[kind="secondary"], button[data-testid="stBaseButton-secondary"]{
  background:#fff; color:var(--mdt-blue-600); border:1.5px solid #D5E1EF; }
.stButton > button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover{
  border-color:var(--mdt-blue); color:var(--mdt-blue); background:#F7FAFD; }
/* Higher-capacity vote buttons: WHITE with a Medtronic cyan->magenta gradient border */
.st-key-pick_a button, .st-key-pick_b button{
  color:var(--mdt-royal) !important; border:1.7px solid transparent !important;
  background:linear-gradient(#fff,#fff) padding-box,
            linear-gradient(135deg,var(--mdt-cyan),var(--mdt-magenta)) border-box !important;
  box-shadow:0 5px 16px rgba(27,15,196,.14) !important; }
.st-key-pick_a button:hover, .st-key-pick_b button:hover{
  transform:translateY(-1px);
  background:linear-gradient(#F6FBFF,#F6FBFF) padding-box,
            linear-gradient(135deg,var(--mdt-royal),var(--mdt-magenta)) border-box !important;
  box-shadow:0 9px 24px rgba(27,15,196,.22) !important; }
.st-key-pick_a button p, .st-key-pick_b button p{ color:var(--mdt-royal) !important; font-weight:700 !important; }

.pcard{ background:linear-gradient(180deg,#F6FAFF 0%,#FFFFFF 132px); border:1px solid var(--line);
  border-radius:18px; padding:18px 20px 20px; min-height:410px; box-shadow:0 4px 18px rgba(16,35,58,.06);
  position:relative; overflow:hidden; transition:box-shadow .18s,transform .18s,border-color .18s; }
.pcard::before{ content:''; position:absolute; top:0; left:0; right:0; height:5px;
  background:linear-gradient(90deg,var(--mdt-royal),var(--mdt-blue),var(--mdt-cyan),var(--mdt-magenta)); }
.pcard:hover{ box-shadow:0 12px 32px rgba(0,104,179,.15); border-color:#CFE0F1; transform:translateY(-2px); }
.pcard-slot{ position:absolute; top:13px; right:15px; width:26px; height:26px; border-radius:8px;
  background:linear-gradient(135deg,var(--mdt-royal),var(--mdt-cyan)); color:#fff; font-weight:800; font-size:13px;
  display:flex; align-items:center; justify-content:center; border:none;
  box-shadow:0 3px 9px rgba(27,15,196,.28); }
.pcard-top{ display:flex; gap:13px; align-items:center; border-bottom:1px solid var(--line);
  padding-bottom:12px; margin-bottom:13px; }
.pcard-avatar{ width:50px; height:50px; border-radius:13px; flex:none;
  background:linear-gradient(135deg,var(--mdt-royal),var(--mdt-cyan)); color:#fff;
  font-weight:800; font-size:15px; display:flex; align-items:center; justify-content:center;
  border:none; box-shadow:0 4px 12px rgba(27,15,196,.22); }
.pcard-demo{ font-size:20px; font-weight:800; color:var(--ink); line-height:1.1; }
.pcard-sub{ font-size:12.5px; color:var(--muted); margin-top:3px; }
.pcard-labs{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; }
.pcard-section{ margin-bottom:11px; }
.pcard-label{ font-size:10.5px; font-weight:700; letter-spacing:.07em; text-transform:uppercase;
  color:var(--mdt-blue-700); margin-bottom:6px; position:relative; padding-left:10px; }
.pcard-label::before{ content:''; position:absolute; left:0; top:1px; bottom:1px; width:3px; border-radius:2px;
  background:linear-gradient(180deg,var(--mdt-cyan),var(--mdt-magenta)); }
.lab-chip{ display:inline-block; padding:5px 10px; border-radius:9px; font-size:12.5px; font-weight:600; }
.lab-good{ background:var(--good-bg); color:var(--good); }
.lab-warn{ background:var(--warn-bg); color:var(--warn); }
.lab-bad{ background:var(--bad-bg); color:var(--bad); }
.lab-neutral{ background:#EDF1F7; color:#3A4453; }
.badge{ display:inline-block; padding:4px 10px; border-radius:20px; font-size:12px; font-weight:500;
  background:#F1F5FA; color:#39465A; border:1px solid #E4EBF3; margin:0 5px 5px 0; }
.badge-none{ font-size:12px; color:#A9B4C4; font-style:italic; }

.vs-wrap{ display:flex; justify-content:center; align-items:center; min-height:410px; height:100%; }
.vs-badge{ width:46px; height:46px; border-radius:50%; color:var(--mdt-royal); font-weight:800;
  font-size:14px; display:flex; align-items:center; justify-content:center; border:2px solid transparent;
  background:linear-gradient(#fff,#fff) padding-box,
            linear-gradient(135deg,var(--mdt-cyan),var(--mdt-magenta)) border-box;
  box-shadow:0 5px 16px rgba(27,15,196,.20); }

.hint-row{ text-align:center; color:var(--muted); font-size:13px; margin-top:10px; }
.kbd{ display:inline-block; min-width:20px; padding:1px 7px; border-radius:6px; border:1px solid var(--line);
  border-bottom-width:2px; background:#fff; font-size:12px; color:var(--ink); }

[data-testid="stMetric"]{ background:var(--card); border:1px solid var(--line); border-radius:14px;
  padding:14px 16px; box-shadow:0 2px 10px rgba(16,35,58,.05); }
[data-testid="stMetricValue"]{ color:var(--mdt-navy); font-weight:800; }
[data-testid="stMetricLabel"] p{ color:var(--muted); font-weight:600; }

[data-testid="stTabs"] [data-baseweb="tab-list"]{ gap:4px; }
[data-testid="stTabs"] [data-baseweb="tab"]{ font-weight:700; color:var(--muted); }
[data-testid="stTabs"] [aria-selected="true"]{ color:var(--mdt-royal) !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{
  background:linear-gradient(90deg,var(--mdt-cyan),var(--mdt-magenta)); }
[data-testid="stProgress"] [role="progressbar"] > div{
  background:linear-gradient(90deg,var(--mdt-cyan),var(--mdt-royal)) !important; }

[data-testid="stSidebar"]{ background:#fff; border-right:1px solid var(--line); }
.side-brand{ display:flex; align-items:center; gap:9px; font-weight:800; color:var(--mdt-navy);
  font-size:15px; margin-bottom:12px; }
.side-id{ display:flex; align-items:center; gap:8px; font-weight:700; color:var(--ink);
  background:var(--mdt-sky); border:1px solid #CFE0F1; border-radius:10px; padding:8px 12px; font-size:14px; }
.side-id .dot{ width:8px; height:8px; border-radius:50%; background:#22B07D; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------- data loads --
@st.cache_data(show_spinner=False)
def load_cases() -> pd.DataFrame:
    df = pd.read_csv(CASES_PATH)
    df["case_id"] = df["case_id"].astype(int)
    return df


def _votes_sig():
    # In DB mode the local CSV never changes, so key the cache on the live vote count instead;
    # this makes progress + the ladder refresh as soon as any vote is written.
    if storage.enabled():
        return ("db", storage.row_count("votes"))
    p = Path(VOTES_PATH)
    return (p.stat().st_mtime, p.stat().st_size) if p.exists() else (0.0, 0)


@st.cache_data(show_spinner=False)
def compute_state(sig, case_ids_tuple):
    """Reload the pool votes and rebuild the Elo ladder. Cached on the votes-file signature so it
    only recomputes when a new vote is written. Votes are restricted to the current pool."""
    votes = engine.filter_pool_votes(engine.load_votes(VOTES_PATH), case_ids_tuple)
    ratings, counts = engine.elo_replay(votes, case_ids_tuple)
    return votes, ratings, counts


# -------------------------------------------------------------------------------- helpers --
def _rng() -> np.random.Generator:
    return np.random.default_rng()


def new_pair(case_ids, ratings, counts, smart, exclude_champion=None, seen_pairs=None):
    """Draw the next pair. ``seen_pairs`` (a set of ``frozenset({a,b})`` this surgeon has already
    judged) is excluded so a rater never re-sees the same match-up until the pool is exhausted."""
    rng = _rng()
    seen_pairs = seen_pairs or set()
    if exclude_champion is not None:  # champion stays; only draw a new challenger
        champ = int(exclude_champion)
        blocked = [champ] + [int(c) for c in case_ids
                             if frozenset((champ, int(c))) in seen_pairs]
        challenger = engine.draw_case(
            rng, case_ids, counts, ratings,
            exclude=blocked, near=ratings.get(champ), smart=smart,
        )
        return champ, challenger
    a = engine.draw_case(rng, case_ids, counts, ratings, smart=smart)
    blocked = [int(a)] + [int(c) for c in case_ids
                          if frozenset((int(a), int(c))) in seen_pairs]
    b = engine.draw_case(rng, case_ids, counts, ratings, exclude=blocked,
                         near=ratings.get(a), smart=smart)
    return a, b


def _surgeon_slice(votes, surgeon_id, surgeon_name):
    """Rows for one surgeon: match on the stable id, falling back to the display name (legacy)."""
    if votes is None or len(votes) == 0:
        return None
    mask = None
    if "surgeon_id" in votes.columns:
        mask = votes["surgeon_id"].astype("string") == str(surgeon_id)
    if mask is None or not mask.any():
        mask = votes["surgeon"].astype("string") == str(surgeon_name)
    return votes[mask]


def surgeon_progress(votes, surgeon_id, surgeon_name):
    """Lifetime comparisons, distinct cases seen, and already-judged pairs for one surgeon."""
    mine = _surgeon_slice(votes, surgeon_id, surgeon_name)
    if mine is None or len(mine) == 0:
        return 0, 0, set()
    mm = mine.dropna(subset=["winner_case_id", "loser_case_id"])
    pairs = {frozenset((int(w), int(l)))
             for w, l in zip(mm["winner_case_id"].astype(int), mm["loser_case_id"].astype(int))}
    seen_ids = (set(pd.concat([mm["winner_case_id"], mm["loser_case_id"]]).astype(int))
                if len(mm) else set())
    # "too close to call" pairs also count as seen, so surgeons aren't shown them again
    if "outcome" in mine.columns:
        ties = mine[mine["outcome"].astype(str).str.lower() == "tie"].dropna(
            subset=["pair_a_id", "pair_b_id"])
        for a, b in zip(ties["pair_a_id"].astype(int), ties["pair_b_id"].astype(int)):
            pairs.add(frozenset((a, b)))
            seen_ids.update((a, b))
    return int(len(mine)), len(seen_ids), pairs


def build_leaderboard(votes, roster_df=None) -> pd.DataFrame:
    """Surgeons ranked by number of comparisons made (all judgements, ties included).
    Returns columns ``surgeon_id``, ``surgeon``, ``comparisons`` sorted high -> low."""
    cols = ["surgeon_id", "surgeon", "comparisons"]
    if votes is None or len(votes) == 0:
        return pd.DataFrame(columns=cols)
    v = votes.copy()
    if "surgeon_id" not in v.columns:
        v["surgeon_id"] = pd.NA
    if "surgeon" not in v.columns:
        v["surgeon"] = pd.NA
    has_id = v["surgeon_id"].notna() & (v["surgeon_id"].astype(str).str.strip() != "")
    key = v["surgeon_id"].astype("object").where(has_id, v["surgeon"])
    g = (v.groupby(key, dropna=False)
         .agg(surgeon=("surgeon", lambda s: s.dropna().iloc[0] if s.notna().any() else ""),
              comparisons=("timestamp", "size"))
         .reset_index(names="surgeon_id"))
    if roster_df is not None and len(roster_df):
        name_map = dict(zip(roster_df["surgeon_id"], roster_df["display_name"]))
        g["surgeon"] = g["surgeon_id"].map(name_map).fillna(g["surgeon"])
    return g.sort_values("comparisons", ascending=False).reset_index(drop=True)


def standing_stat_html(votes, roster_df=None) -> str:
    """Header stat card: the signed-in surgeon's leaderboard rank and gap to the top."""
    if not st.session_state.get("surgeon"):
        return ""
    lb = build_leaderboard(votes, roster_df)
    me = str(st.session_state.get("surgeon_id") or "")
    pos = lb.index[lb["surgeon_id"].astype(str) == me] if len(lb) else []
    if not len(pos):
        return ""
    r = int(pos[0])
    top = int(lb.iloc[0]["comparisons"])
    mine = int(lb.iloc[r]["comparisons"])
    if r == 0:
        lead = top - int(lb.iloc[1]["comparisons"]) if len(lb) > 1 else 0
        value = "\U0001f947 #1"
        label = f"{lead} ahead of 2nd" if lead else "leading the board"
    else:
        leader = str(lb.iloc[0]["surgeon"]) or str(lb.iloc[0]["surgeon_id"])
        value = f"#{r + 1}"
        label = f"{top - mine} behind {leader}"
    return (f'<div class="hstat"><div class="v">{html.escape(value)}</div>'
            f'<div class="l">{html.escape(label)}</div></div>')


def record_vote(winner, loser, pair_a, pair_b):
    engine.append_vote(
        VOTES_PATH,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "surgeon": st.session_state.surgeon,
            "surgeon_id": st.session_state.surgeon_id,
            "session_id": st.session_state.session_id,
            "outcome": "pick",
            "winner_case_id": int(winner),
            "loser_case_id": int(loser),
            "pair_a_id": int(pair_a),
            "pair_b_id": int(pair_b),
        },
    )
    st.session_state.my_votes += 1


def record_tie(pair_a, pair_b):
    """Record a "too close to call" judgement: the two cases are of equal osteogenic capacity.
    Stored with no winner/loser so the ladder can treat it as a draw."""
    engine.append_vote(
        VOTES_PATH,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "surgeon": st.session_state.surgeon,
            "surgeon_id": st.session_state.surgeon_id,
            "session_id": st.session_state.session_id,
            "outcome": "tie",
            "winner_case_id": None,
            "loser_case_id": None,
            "pair_a_id": int(pair_a),
            "pair_b_id": int(pair_b),
        },
    )
    st.session_state.my_votes += 1


def record_flag(case_id, reason="unrealistic"):
    """Record that the signed-in surgeon judged a case clinically unrealistic / not seen in
    practice -- a face-validity signal on the synthetic cases (not counted as a comparison)."""
    engine.append_flag(
        FLAGS_PATH,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "surgeon": st.session_state.surgeon,
            "surgeon_id": st.session_state.surgeon_id,
            "session_id": st.session_state.session_id,
            "case_id": int(case_id),
            "reason": reason,
        },
    )


def advance_after_vote(winner, loser, case_ids, ratings, counts, smart, seen_pairs=None):
    """Winner-stays / loser-replaced, with champion retirement after a win streak."""
    a, b = st.session_state.pair
    if st.session_state.streak_id == winner:
        st.session_state.streak_count += 1
    else:
        st.session_state.streak_id = winner
        st.session_state.streak_count = 1

    if st.session_state.streak_count >= STREAK_RETIRE:
        st.session_state.pair = new_pair(case_ids, ratings, counts, smart, seen_pairs=seen_pairs)
        st.session_state.streak_id = None
        st.session_state.streak_count = 0
    else:
        _, challenger = new_pair(case_ids, ratings, counts, smart,
                                 exclude_champion=winner, seen_pairs=seen_pairs)
        # keep the winner in whichever slot it occupied
        st.session_state.pair = (winner, challenger) if winner == a else (challenger, winner)


# ------------------------------------------------------------------------------- chrome ---
def render_header(my_comparisons, my_cases_seen, total_votes, rho, is_admin=False, standing_html=""):
    brand = _brand_lockup(dark=True)
    stats = (f'<div class="hstat"><div class="v">{my_comparisons}</div>'
             f'<div class="l">Your comparisons</div></div>'
             f'<div class="hstat"><div class="v">{my_cases_seen}</div>'
             f'<div class="l">Cases seen</div></div>')
    stats += standing_html
    if is_admin:  # collection totals + truth correlation stay on the owner view only
        rho_html = f"{rho:.2f}" if rho is not None else "&mdash;"
        stats += (f'<div class="hstat"><div class="v">{total_votes}</div>'
                  f'<div class="l">Total collected</div></div>'
                  f'<div class="hstat"><div class="v">{rho_html}</div>'
                  f'<div class="l">Rank vs truth</div></div>')
    st.markdown(
        f"""
<div class="app-header">
  <div class="brand">{brand}
    <span class="brand-sep">|</span>
    <span class="app-title">Osteogenic Capacity Study</span></div>
  <div class="header-right">{stats}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_roster_admin():
    """Owner-only roster manager: add surgeons (auto-PIN), reset PINs, activate/remove."""
    with st.expander("Roster & PINs", expanded=False):
        rdf = roster.load_roster(ROSTER_PATH)
        st.caption("Surgeons sign in with their name + 4-digit PIN. PINs live in the "
                   "git-ignored roster.csv — share each surgeon their PIN privately.")
        if len(rdf):
            show = rdf[["display_name", "surgeon_id", "pin", "active"]].rename(
                columns={"display_name": "Surgeon", "surgeon_id": "ID",
                         "pin": "PIN", "active": "Active"})
            st.dataframe(show, hide_index=True, width="stretch")
        na1, na2 = st.columns([3, 1])
        with na1:
            new_name = st.text_input("Add surgeon", key="roster_add",
                                     placeholder="Dr. Name", label_visibility="collapsed")
        with na2:
            if st.button("Add", width="stretch"):
                if new_name.strip():
                    _, row = roster.add_surgeon(ROSTER_PATH, new_name)
                    st.session_state.pending_toast = (
                        f"Added {row['display_name']} · PIN {row['pin']}")
                    st.rerun()
        if len(rdf):
            sel = st.selectbox("Manage surgeon", rdf["display_name"].tolist(), key="roster_sel",
                               index=None, placeholder="Manage a surgeon…",
                               label_visibility="collapsed")
            if sel:
                sid = rdf.loc[rdf["display_name"] == sel, "surgeon_id"].iloc[0]
                active_now = int(rdf.loc[rdf["surgeon_id"] == sid, "active"].iloc[0]) == 1
                m1, m2, m3 = st.columns(3)
                with m1:
                    if st.button("Reset PIN", width="stretch"):
                        pin = roster.reset_pin(ROSTER_PATH, sid)
                        st.session_state.pending_toast = f"{sel}: new PIN {pin}"
                        st.rerun()
                with m2:
                    if st.button("Deactivate" if active_now else "Activate", width="stretch"):
                        roster.set_active(ROSTER_PATH, sid, not active_now)
                        st.rerun()
                with m3:
                    if st.button("Remove", width="stretch"):
                        roster.remove_surgeon(ROSTER_PATH, sid)
                        st.rerun()


def render_welcome(roster_df):
    hero_logo = (f"<span class='hero-badge'><img src='{LOGO_URI}' class='hero-logo' alt='Medtronic'/>"
                 f"</span>" if LOGO_URI else "")
    st.markdown(
        f"""
<div class="hero">
  {hero_logo}
  <h1>Which patient can grow more bone?</h1>
  <p>You'll see two patient profiles side by side. Pick the one with the higher
     <b>osteogenic capacity</b> &mdash; the patient's intrinsic bone-forming ability. Each choice
     takes a second, and together they build a shared ranking that trains the model.</p>
  <div class="hero-steps">
    <div class="hero-step"><div class="n">STEP 01</div><div class="t">Compare</div>
      <div class="d">Two real-world profiles, side by side.</div></div>
    <div class="hero-step"><div class="n">STEP 02</div><div class="t">Choose</div>
      <div class="d">Click the stronger healer &mdash; or tap &larr; / &rarr;.</div></div>
    <div class="hero-step"><div class="n">STEP 03</div><div class="t">Repeat</div>
      <div class="d">The winner stays; a fresh challenger appears.</div></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown("<div class='sec-title'>Sign in to begin</div>", unsafe_allow_html=True)
    active = roster.active_surgeons(roster_df)
    if len(active) == 0:
        st.warning("No surgeons on the roster yet. Ask the study owner to add you.")
        return
    names = active["display_name"].tolist()
    c1, c2, c3 = st.columns([3, 1.5, 1.3])
    with c1:
        pick = st.selectbox("Surgeon", names, key="welcome_pick", index=None,
                            placeholder="Select your name", label_visibility="collapsed")
    with c2:
        pin = st.text_input("PIN", key="welcome_pin", type="password", max_chars=4,
                           placeholder="4-digit PIN", label_visibility="collapsed")
    with c3:
        start = st.button("Start  →", type="primary", width="stretch")
    st.caption("Your PIN keeps each surgeon's comparisons separate. Ask the study owner if you "
               "don't have one yet.")
    if start:
        if not pick:
            st.warning("Please select your name to begin.")
        else:
            sid = active.loc[active["display_name"] == pick, "surgeon_id"].iloc[0]
            if roster.verify_pin(roster_df, sid, pin or ""):
                st.session_state.surgeon = pick
                st.session_state.surgeon_id = str(sid)
                st.session_state.session_started_at = time.time()
                st.session_state.my_votes = 0
                st.session_state.pair = None
                st.rerun()
            else:
                st.error("Incorrect PIN. Try again or contact the study owner.")


def inject_keyboard():
    # Install a single persistent keydown listener into the PARENT document so it survives
    # Streamlit reruns (and iframe teardown). Guarded so components.html runs once per session,
    # which keeps the terminal clean of repeated deprecation notices.
    if st.session_state.get("_kb_installed"):
        return
    st.session_state["_kb_installed"] = True
    components.html(
        """
<script>
const pdoc = window.parent.document;
if (!pdoc.__capKeyInstalled) {
  pdoc.__capKeyInstalled = true;
  const s = pdoc.createElement('script');
  s.textContent = [
    "(function(){",
    "  function clickMark(mark){",
    "    var btns = document.querySelectorAll('button');",
    "    for (var i=0;i<btns.length;i++){ var b=btns[i];",
    "      if(b.innerText && b.innerText.indexOf(mark)!==-1){ b.click(); return; } }",
    "  }",
    "  document.addEventListener('keydown', function(e){",
    "    var ae = document.activeElement;",
    "    if(ae && (ae.tagName==='INPUT' || ae.tagName==='TEXTAREA')) return;",
    "    if(e.repeat) return;",
    "    if(e.key==='ArrowLeft' || e.key==='1'){ e.preventDefault(); clickMark('◀'); }",
    "    else if(e.key==='ArrowRight' || e.key==='2'){ e.preventDefault(); clickMark('▶'); }",
    "    else if(e.key==='ArrowDown' || e.key===' '){ e.preventDefault(); clickMark('Too close'); }",
    "  });",
    "})();"
  ].join(' ');
  pdoc.head.appendChild(s);
}
</script>
""",
        height=0,
    )



# ============================================================================ sidebar ======
cases = load_cases()
pool_ids = engine.load_or_make_pool(OUT_DIR, cases["case_id"].tolist(), n=POOL_SIZE, seed=POOL_SEED)
case_ids = pool_ids  # the game now runs on the reduced, fixed pool
id_to_row = {int(r.case_id): r for r in cases.itertuples(index=False)}
roster_df = roster.seed_if_empty(ROSTER_PATH)

st.session_state.setdefault("session_id", uuid.uuid4().hex[:12])
st.session_state.setdefault("surgeon", "")
st.session_state.setdefault("surgeon_id", "")
st.session_state.setdefault("session_started_at", None)
st.session_state.setdefault("pair", None)
st.session_state.setdefault("streak_id", None)
st.session_state.setdefault("streak_count", 0)
st.session_state.setdefault("my_votes", 0)
st.session_state.setdefault("pending_toast", None)
st.session_state.setdefault("is_admin", False)

is_admin = bool(st.session_state.is_admin)  # owner (insights/exports) vs surgeon (compare only)

with st.sidebar:
    st.markdown(f"<div class='side-brand'>{_brand_lockup(dark=False, cls='side-logo')}"
                f"<span>Capacity Study</span></div>", unsafe_allow_html=True)
    if st.session_state.surgeon:
        st.markdown(f"<div class='side-id'><span class='dot'></span>{st.session_state.surgeon}</div>",
                    unsafe_allow_html=True)
        if st.button("Sign out", width="stretch"):
            st.session_state.surgeon = ""
            st.session_state.surgeon_id = ""
            st.session_state.session_started_at = None
            st.session_state.my_votes = 0
            st.session_state.pair = None
            st.rerun()
    if is_admin:  # owner-only controls; hidden from surgeons
        st.divider()
        st.markdown("##### Settings")
        n_buckets = st.selectbox("Capacity buckets", [3, 5, 10], index=1)
        smart = st.toggle(
            "Smart pairing", value=True,
            help="Favour under-sampled cases and near-equal match-ups for faster, more informative "
                 "ranking. Turn off for purely random challengers.",
        )
        st.caption(f"Champion retires after {STREAK_RETIRE} straight wins · "
                   f"pool of {len(case_ids)} cases.")
        render_roster_admin()
    else:
        n_buckets, smart = 5, True

votes, ratings, counts = compute_state(_votes_sig(), case_ids)
ratings_df = engine.ratings_dataframe(ratings, counts)
n_seen = int((ratings_df["n_compares"] > 0).sum())
truth = analysis.spearman_vs_truth(ratings_df, cases)
rho = truth[0] if truth else None
my_comp, my_seen, seen_pairs = surgeon_progress(
    votes, st.session_state.surgeon_id, st.session_state.surgeon)

with st.sidebar:
    st.divider()
    st.markdown("##### Your progress")
    if st.session_state.surgeon:
        pc1, pc2 = st.columns(2)
        pc1.metric("Comparisons", my_comp)
        pc2.metric("Cases seen", f"{my_seen}/{len(case_ids)}")
        nxt = next((m for m in MILESTONES if m > my_comp), None)
        if nxt:
            st.progress(min(my_comp / nxt, 1.0), text=f"{my_comp} · next milestone {nxt}")
        else:
            st.caption(f"🏅 {my_comp} comparisons — outstanding contribution!")
        started = st.session_state.get("session_started_at")
        if started and (time.time() - started) >= SESSION_MINUTES * 60:
            st.caption(f"You've been going ~{SESSION_MINUTES}+ min — a great batch. "
                       "Keep going or take a break; every answer is saved.")
    else:
        st.caption("Sign in to track your progress.")

    st.markdown("##### \U0001f3c6 Leaderboard")
    lb = build_leaderboard(votes, roster_df)
    if len(lb) == 0:
        st.caption("No comparisons yet \u2014 be the first on the board!")
    else:
        medals = {0: "\U0001f947", 1: "\U0001f948", 2: "\U0001f949"}
        me_id = str(st.session_state.get("surgeon_id") or "")
        me_name = str(st.session_state.get("surgeon") or "")
        top = lb.head(10)
        lines = []
        for i, row in top.iterrows():
            badge = medals.get(i, f"{i + 1}.")
            name = str(row["surgeon"]) or str(row["surgeon_id"])
            is_me = (me_id and str(row["surgeon_id"]) == me_id) or (me_name and name == me_name)
            label = f"**{name} (you)**" if is_me else name
            lines.append(f"{badge} {label} \u2014 {int(row['comparisons'])}")
        st.markdown("<br>".join(lines), unsafe_allow_html=True)
        # if the signed-in surgeon isn't in the top 10, show their standing too
        shown = set(top["surgeon_id"].astype(str))
        if me_id and me_id not in shown:
            pos = lb.index[lb["surgeon_id"].astype(str) == me_id]
            if len(pos):
                r = int(pos[0])
                st.caption(f"\u22ef {r + 1}. {me_name} (you) \u2014 "
                           f"{int(lb.iloc[r]['comparisons'])}")

    if is_admin:  # dataset coverage + truth diagnostics are owner-only
        st.markdown("##### Dataset coverage")
        st.progress(min(n_seen / max(len(case_ids), 1), 1.0),
                    text=f"{n_seen} / {len(case_ids)} cases sampled")
        st.caption(f"{len(votes)} comparisons collected · "
                   f"{votes['surgeon'].nunique() if len(votes) else 0} surgeon(s)")
        if rho is not None:
            st.caption(f"Live ladder vs hidden truth: Spearman ρ = {rho:.2f}")

    # ---- owner access ----------------------------------------------------------------
    # The passcode box is revealed only via ?owner=1, so the link you send to surgeons
    # shows nothing but the comparison task. The passcode itself is the real gate.
    st.divider()
    if is_admin:
        st.success("Owner mode — insights unlocked")
        if st.button("Exit owner mode", width="stretch"):
            st.session_state.is_admin = False
            try:
                st.query_params.clear()
            except Exception:
                pass
            st.rerun()
    elif "owner" in st.query_params:
        with st.expander("Owner access", expanded=True):
            if _admin_passcode() is None:
                st.caption("Set `admin_passcode` in `.streamlit/secrets.toml` to enable insights.")
            entered = st.text_input("Owner passcode", type="password", key="admin_pc",
                                    label_visibility="collapsed", placeholder="Owner passcode")
            if st.button("Unlock insights", width="stretch",
                         disabled=_admin_passcode() is None):
                if _passcode_ok(entered):
                    st.session_state.is_admin = True
                    st.rerun()
                else:
                    st.error("Incorrect passcode.")

# ============================================================================ layout =======
if st.session_state.pending_toast:
    st.toast(st.session_state.pending_toast)
    st.session_state.pending_toast = None

render_header(my_comp, my_seen, len(votes), rho, is_admin,
              standing_html=standing_stat_html(votes, roster_df))

if not st.session_state.surgeon:
    render_welcome(roster_df)
    st.stop()

if is_admin:
    compare_tab, insights_tab = st.tabs(["\u2003Compare\u2003", "\u2003Insights\u2003"])
else:
    # surgeons get a single clean Compare view (no tabs, no Insights)
    compare_tab, insights_tab = contextlib.nullcontext(), contextlib.nullcontext()

# --------------------------------------------------------------------------------- COMPARE -
with compare_tab:
    st.markdown("<div class='sec-title'>Which patient has the higher osteogenic capacity?</div>",
                unsafe_allow_html=True)
    st.caption("Osteogenic capacity = the patient's intrinsic bone-forming ability. "
               "Judge the biology; ignore the planned surgery.")

    if st.session_state.pair is None:
        st.session_state.pair = new_pair(case_ids, ratings, counts, smart, seen_pairs=seen_pairs)
    id_a, id_b = st.session_state.pair
    row_a, row_b = id_to_row[int(id_a)], id_to_row[int(id_b)]

    left, mid, right = st.columns([10, 1, 10])
    with left:
        st.markdown(case_display.case_card_html(row_a._asdict(), "A"), unsafe_allow_html=True)
        pick_a = st.button("◀  Higher capacity", key="pick_a", width="stretch", type="primary")
        flag_a = st.button("\U0001f6a9 Unrealistic case", key="flag_a", width="stretch",
                           help="Flag this patient as clinically unrealistic / not seen in practice.")
    with mid:
        st.markdown("<div class='vs-wrap'><div class='vs-badge'>VS</div></div>",
                    unsafe_allow_html=True)
    with right:
        st.markdown(case_display.case_card_html(row_b._asdict(), "B"), unsafe_allow_html=True)
        pick_b = st.button("Higher capacity  ▶", key="pick_b", width="stretch", type="primary")
        flag_b = st.button("\U0001f6a9 Unrealistic case", key="flag_b", width="stretch",
                           help="Flag this patient as clinically unrealistic / not seen in practice.")

    s1, s2, s3 = st.columns([3, 2, 3])
    with s2:
        skip = st.button("Too close to call", key="skip", width="stretch")
    st.markdown(
        "<div class='hint-row'>Use <span class='kbd'>&larr;</span> / "
        "<span class='kbd'>&rarr;</span> to choose &nbsp;·&nbsp; "
        "<span class='kbd'>&darr;</span> tie</div>",
        unsafe_allow_html=True,
    )
    inject_keyboard()

    if pick_a or pick_b:
        winner = int(id_a) if pick_a else int(id_b)
        loser = int(id_b) if pick_a else int(id_a)
        record_vote(winner, loser, id_a, id_b)
        pairs_now = set(seen_pairs) | {frozenset((winner, loser))}
        advance_after_vote(winner, loser, case_ids, ratings, counts, smart, pairs_now)
        with contextlib.suppress(Exception):  # refresh model-ready exports on every vote
            exports.regenerate(OUT_DIR, engine.load_votes(VOTES_PATH), cases, case_ids, roster_df)
        lifetime = my_comp + 1
        if lifetime in MILESTONES:
            st.session_state.pending_toast = f"🎯 {lifetime} comparisons — thank you!"
        else:
            st.session_state.pending_toast = "✓ Recorded"
        st.rerun()

    if skip:
        record_tie(id_a, id_b)
        pairs_now = set(seen_pairs) | {frozenset((int(id_a), int(id_b)))}
        st.session_state.pair = new_pair(case_ids, ratings, counts, smart, seen_pairs=pairs_now)
        st.session_state.streak_id, st.session_state.streak_count = None, 0
        with contextlib.suppress(Exception):  # refresh model-ready exports on every judgement
            exports.regenerate(OUT_DIR, engine.load_votes(VOTES_PATH), cases, case_ids, roster_df)
        st.session_state.pending_toast = "\u2713 Tie recorded"
        st.rerun()

    if flag_a or flag_b:
        flagged = int(id_a) if flag_a else int(id_b)
        kept = int(id_b) if flag_a else int(id_a)
        record_flag(flagged)
        _, challenger = new_pair(case_ids, ratings, counts, smart,
                                 exclude_champion=kept, seen_pairs=seen_pairs)
        st.session_state.pair = (challenger, kept) if flag_a else (kept, challenger)
        st.session_state.streak_id, st.session_state.streak_count = None, 0
        st.session_state.pending_toast = "\U0001f6a9 Flagged \u2014 new case loaded"
        st.rerun()

# ------------------------------------------------------------------------------ INSIGHTS --
with insights_tab:
    if not is_admin:
        st.stop()  # insights + data exports are owner-only; surgeons only compare
    if len(votes) == 0:
        st.info("No comparisons yet — head to the Compare tab to build the ranking.")
    else:
        buckets_df = engine.assign_buckets(ratings_df, n_buckets=n_buckets)
        labels = engine.default_labels(n_buckets)

        st.markdown("<div class='sec-title'>Collection overview</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Comparisons", len(votes))
        c2.metric("Cases ranked", f"{n_seen} / {len(case_ids)}")
        c3.metric("Surgeons", votes["surgeon"].nunique())
        agree = analysis.inter_surgeon_agreement(votes)
        c4.metric("Surgeon agreement",
                  f"{agree[0]*100:.0f}%" if agree else "n/a",
                  help=None if agree else "Needs pairs judged by 2+ surgeons.")

        flags = engine.load_flags(FLAGS_PATH)
        st.markdown("<div class='sec-title'>Realism flags</div>", unsafe_allow_html=True)
        if len(flags) == 0:
            st.caption("No cases have been flagged as clinically unrealistic yet.")
        else:
            fl = flags.dropna(subset=["case_id"]).copy()
            fl["case_id"] = fl["case_id"].astype(int)
            summ = (fl.groupby("case_id")
                      .agg(times_flagged=("case_id", "size"), surgeons=("surgeon_id", "nunique"))
                      .reset_index()
                      .sort_values(["times_flagged", "case_id"], ascending=[False, True]))
            summ["patient"] = summ["case_id"].map(
                lambda cid: case_display.short_summary(id_to_row[int(cid)]._asdict())
                if int(cid) in id_to_row else f"case {cid}")
            fc1, fc2 = st.columns(2)
            fc1.metric("Flagged cases", int(summ["case_id"].nunique()))
            fc2.metric("Total flags", int(len(fl)))
            st.dataframe(summ[["case_id", "patient", "times_flagged", "surgeons"]],
                         hide_index=True, width="stretch")

        merged = buckets_df.merge(cases, on="case_id", how="left")
        merged["patient"] = merged.apply(lambda r: case_display.short_summary(r), axis=1)
        show_cols = ["case_id", "patient", "rating", "n_compares", "bucket"]
        ranked = merged[merged["bucket"] != "Unranked"]

        st.markdown("<div class='sec-title'>Capacity buckets</div>", unsafe_allow_html=True)
        dist = (ranked.groupby("bucket").size().reindex(labels, fill_value=0)
                .rename_axis("bucket").reset_index(name="cases"))
        chart = (
            alt.Chart(dist)
            .mark_bar(color="#0068B3", cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(
                x=alt.X("bucket:N", sort=labels, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("cases:Q", title="cases"),
                tooltip=["bucket", "cases"],
            )
            .properties(height=260, width="container")
        )
        st.altair_chart(chart)

        lb1, lb2 = st.columns(2)
        with lb1:
            st.markdown("<div class='sec-title'>Top capacity</div>", unsafe_allow_html=True)
            st.dataframe(ranked.nlargest(12, "rating")[show_cols],
                         hide_index=True, width="stretch")
        with lb2:
            st.markdown("<div class='sec-title'>Lowest capacity</div>", unsafe_allow_html=True)
            st.dataframe(ranked.nsmallest(12, "rating")[show_cols],
                         hide_index=True, width="stretch")

        with st.expander("Bradley-Terry cross-check (batch model)"):
            st.caption(
                "A regularised Bradley-Terry fit of latent strengths from the full win/loss "
                "graph — a more principled cross-check of the online Elo ladder."
            )
            if st.button("Fit Bradley-Terry"):
                with st.spinner("Fitting..."):
                    bt = analysis.bradley_terry(votes, case_ids)
                if bt is None or len(bt) < 3:
                    st.warning("Not enough comparisons yet.")
                else:
                    elo_bt = ratings_df.merge(bt, on="case_id")
                    from scipy.stats import spearmanr
                    rho_eb = spearmanr(elo_bt["rating"], elo_bt["bt_strength"]).statistic
                    m1, m2 = st.columns(2)
                    m1.metric("Elo vs Bradley-Terry", f"{rho_eb:.2f}")
                    bt_truth = analysis.spearman_vs_truth(
                        bt.assign(n_compares=1), cases, score_col="bt_strength"
                    )
                    m2.metric("Bradley-Terry vs truth",
                              f"{bt_truth[0]:.2f}" if bt_truth else "n/a")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div class='sec-title'>Per-surgeon contribution</div>",
                        unsafe_allow_html=True)
            st.dataframe(analysis.surgeon_summary(votes), hide_index=True, width="stretch")
        with col_b:
            st.markdown("<div class='sec-title'>Model-ready exports</div>",
                        unsafe_allow_html=True)
            st.caption(f"Auto-written to `{OUT_DIR.relative_to(ROOT)}` on every vote — tidy, "
                       "surgeon-normalised tables ready for the capacity model.")
            with contextlib.suppress(Exception):
                exports.regenerate(OUT_DIR, votes, cases, case_ids, roster_df)
            files = [
                ("case_labels.csv", "Per-case score + tier + features (+ owner-only truth)"),
                ("comparisons.csv", "Tidy pairwise judgements"),
                ("surgeons.csv", "Per-surgeon contribution + reliability"),
                ("cases.csv", "Pool case features"),
            ]
            for fname, desc in files:
                fp = OUT_DIR / fname
                if fp.exists():
                    st.download_button(fname, fp.read_bytes(), fname, "text/csv",
                                       width="stretch", help=desc)
