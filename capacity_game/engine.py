"""Core game logic for the osteogenic-capacity comparison game.

Pure Python (no Streamlit) so it can be unit-tested and reused from notebooks.

Design
------
* ``votes.csv`` is the append-only **source of truth**. Every surgeon click writes one row.
* A global **Elo** ladder is reconstructed by replaying the votes in timestamp order. Elo is
  online, always defined, and robust to sparse / unbalanced comparison graphs -- ideal for a
  live "which case has higher capacity?" game.
* Cases are ranked by their Elo rating and split into equal-size **buckets** (quantiles),
  giving the least -> most capacity ordering the model can train on.
"""

from __future__ import annotations

import csv
import threading
from pathlib import Path

import numpy as np
import pandas as pd

try:  # optional durable SQL backend for ephemeral cloud hosts (see storage.py)
    import storage
except Exception:  # pragma: no cover - storage backend is optional
    storage = None

# --- Elo constants -----------------------------------------------------------------------
BASE_RATING = 1500.0  # every case starts here
K_FACTOR = 32.0       # step size of each update (standard Elo default)

# --- votes.csv schema (append-only log) --------------------------------------------------
VOTE_COLS = [
    "timestamp",        # ISO-8601 UTC
    "surgeon",          # display name of the rater (from the roster)
    "surgeon_id",       # stable roster id (sNNN) -- authoritative key for per-surgeon tracking
    "session_id",       # browser session (uuid)
    "outcome",          # "pick" (a strict preference) or "tie" (too close to call)
    "winner_case_id",   # case judged to have HIGHER osteogenic capacity (blank for a tie)
    "loser_case_id",    # the other case (blank for a tie)
    "pair_a_id",        # left card shown (audit)
    "pair_b_id",        # right card shown (audit)
]

_ID_COLS = ("winner_case_id", "loser_case_id", "pair_a_id", "pair_b_id")

# --- flags.csv schema: cases a surgeon marks as clinically unrealistic (face-validity) ----
FLAG_COLS = [
    "timestamp",    # ISO-8601 UTC
    "surgeon",      # display name of the rater
    "surgeon_id",   # stable roster id (sNNN)
    "session_id",   # browser session (uuid)
    "case_id",      # the case flagged as unrealistic / not seen in the real world
    "reason",       # free-text / tag (default "unrealistic")
]

# Historical schema (pre-``surgeon_id``); used to migrate old logs in place.
_LEGACY_VOTE_COLS = [
    "timestamp", "surgeon", "session_id",
    "winner_case_id", "loser_case_id", "pair_a_id", "pair_b_id",
]

# Serialises appends so concurrent surgeon sessions (one shared server process) can't interleave
# writes to votes.csv. A single-instance deployment is assumed; scale-out would need a real store.
_APPEND_LOCK = threading.Lock()


# ============================================================================ persistence ==
def votes_path(out_dir) -> Path:
    return Path(out_dir) / "votes.csv"


def flags_path(out_dir) -> Path:
    return Path(out_dir) / "flags.csv"


def _ensure_schema(path) -> None:
    """Upgrade a legacy / ragged votes.csv to the current ``VOTE_COLS`` schema in place.

    Old logs were written before ``surgeon_id`` existed (7 columns). Appending new 8-column
    rows to such a file corrupts it (ragged rows). This rewrites every row into the current
    schema, mapping each row by its width, so loading and appending are always consistent.
    """
    path = Path(path)
    if not (path.exists() and path.stat().st_size > 0):
        return
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] == VOTE_COLS:
        return  # already current
    out = [VOTE_COLS]
    for row in rows[1:]:
        if len(row) == len(VOTE_COLS):
            rec = dict(zip(VOTE_COLS, row))
        elif len(row) == len(_LEGACY_VOTE_COLS):
            rec = dict(zip(_LEGACY_VOTE_COLS, row))       # surgeon_id absent -> ""
        else:
            rec = dict(zip(rows[0], row))                 # best-effort by original header
        out.append([rec.get(c, "") for c in VOTE_COLS])
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(out)


def load_votes(path) -> pd.DataFrame:
    """Read the append-only vote log, returning an empty typed frame if it does not exist.
    Legacy logs are migrated to the current schema first; missing columns are backfilled NA.
    When a durable SQL backend is configured (deployment) votes are read from it instead."""
    if storage is not None and storage.enabled():
        df = storage.load_table("votes", VOTE_COLS)
        for c in _ID_COLS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        return df
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        _ensure_schema(path)
        df = pd.read_csv(path)
        for c in _ID_COLS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        for c in VOTE_COLS:
            if c not in df.columns:
                df[c] = pd.NA
        return df
    return pd.DataFrame(columns=VOTE_COLS)


def append_vote(path, row: dict) -> None:
    """Append a single vote row, writing the header only when creating the file. Any legacy
    on-disk schema is migrated first so appended rows never go ragged. A process-wide lock
    serialises concurrent surgeon sessions appending to the same log. When a durable SQL
    backend is configured (deployment) the row is appended to it instead."""
    if storage is not None and storage.enabled():
        storage.append_row("votes", row, VOTE_COLS)
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _APPEND_LOCK:
        _ensure_schema(path)
        write_header = not path.exists() or path.stat().st_size == 0
        pd.DataFrame([{c: row.get(c) for c in VOTE_COLS}], columns=VOTE_COLS).to_csv(
            path, mode="a", header=write_header, index=False
        )


def load_flags(path) -> pd.DataFrame:
    """Read the realism-flag log (cases marked clinically unrealistic). Storage-backed in
    deployment; local CSV otherwise; empty typed frame when nothing has been flagged yet."""
    if storage is not None and storage.enabled():
        df = storage.load_table("flags", FLAG_COLS)
    else:
        path = Path(path)
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_csv(path)
            for c in FLAG_COLS:
                if c not in df.columns:
                    df[c] = pd.NA
            df = df[FLAG_COLS]
        else:
            return pd.DataFrame(columns=FLAG_COLS)
    if "case_id" in df.columns:
        df["case_id"] = pd.to_numeric(df["case_id"], errors="coerce").astype("Int64")
    return df


def append_flag(path, row: dict) -> None:
    """Append one realism-flag row. Storage-backed in deployment; local CSV otherwise."""
    if storage is not None and storage.enabled():
        storage.append_row("flags", row, FLAG_COLS)
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _APPEND_LOCK:
        write_header = not path.exists() or path.stat().st_size == 0
        pd.DataFrame([{c: row.get(c) for c in FLAG_COLS}], columns=FLAG_COLS).to_csv(
            path, mode="a", header=write_header, index=False
        )


# ================================================================================= pool ====
def pool_path(out_dir) -> Path:
    return Path(out_dir) / "case_pool.csv"


def load_or_make_pool(out_dir, case_ids, n: int = 200, seed: int = 42) -> tuple:
    """Load the fixed comparison pool, creating a seeded random sample on first run.

    Reducing the full case set to a stable ~n-case pool makes it realistic for surgeons to
    cover every case, while a uniform random draw keeps the pool representative. The pool is
    persisted so it stays identical across sessions, surgeons and machines.
    """
    path = pool_path(out_dir)
    all_ids = [int(c) for c in case_ids]
    all_set = set(all_ids)
    if path.exists() and path.stat().st_size > 0:
        pool = [int(c) for c in pd.read_csv(path)["case_id"].tolist() if int(c) in all_set]
        if pool:
            return tuple(pool)
    n = min(n, len(all_ids))
    rng = np.random.default_rng(seed)
    pool = sorted(int(c) for c in rng.choice(all_ids, size=n, replace=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"case_id": pool}).to_csv(path, index=False)
    return tuple(pool)


def filter_pool_votes(votes: pd.DataFrame, pool_ids) -> pd.DataFrame:
    """Keep only votes whose winner AND loser are both in the pool.

    Legacy / off-pool votes are dropped so the ladder and exports reflect the current pool.
    """
    if votes is None or len(votes) == 0:
        return votes
    pool = {int(c) for c in pool_ids}
    v = votes.copy()
    has_outcome = "outcome" in v.columns
    is_tie = ((v["outcome"].astype(str).str.lower() == "tie")
              if has_outcome else pd.Series(False, index=v.index))

    picks = v[~is_tie].dropna(subset=["winner_case_id", "loser_case_id"])
    if len(picks):
        picks = picks[picks["winner_case_id"].astype(int).isin(pool)
                      & picks["loser_case_id"].astype(int).isin(pool)]
    ties = v[is_tie].dropna(subset=["pair_a_id", "pair_b_id"]) if has_outcome else v.iloc[0:0]
    if len(ties):
        ties = ties[ties["pair_a_id"].astype(int).isin(pool)
                    & ties["pair_b_id"].astype(int).isin(pool)]
    out = pd.concat([picks, ties]) if len(ties) else picks
    return out.reset_index(drop=True)


# ================================================================================== elo ====
def elo_replay(votes: pd.DataFrame, case_ids, k: float = K_FACTOR, base: float = BASE_RATING):
    """Reconstruct Elo ratings + comparison counts by replaying votes in timestamp order.

    Deterministic for fixed ``k`` / ``base`` -> the same log always yields the same ladder.

    Returns
    -------
    (ratings, counts) : two dicts keyed by ``case_id``.
    """
    ratings = {int(c): base for c in case_ids}
    counts = {int(c): 0 for c in case_ids}

    if votes is None or len(votes) == 0:
        return ratings, counts

    v = votes.copy()
    if "timestamp" in v.columns:
        v = v.sort_values("timestamp", kind="stable")
    has_outcome = "outcome" in v.columns

    for row in v.itertuples(index=False):
        tie = has_outcome and str(getattr(row, "outcome", "")).lower() == "tie"
        if tie:  # a "too close to call" -> draw: both cases score 0.5
            a, b = getattr(row, "pair_a_id"), getattr(row, "pair_b_id")
            if pd.isna(a) or pd.isna(b):
                continue
            i, j, si = int(a), int(b), 0.5
        else:
            w, l = getattr(row, "winner_case_id"), getattr(row, "loser_case_id")
            if pd.isna(w) or pd.isna(l):
                continue
            i, j, si = int(w), int(l), 1.0
        if i not in ratings:
            ratings[i], counts[i] = base, 0
        if j not in ratings:
            ratings[j], counts[j] = base, 0
        ri, rj = ratings[i], ratings[j]
        ei = 1.0 / (1.0 + 10.0 ** ((rj - ri) / 400.0))  # expected score for i
        ratings[i] = ri + k * (si - ei)
        ratings[j] = rj + k * ((1.0 - si) - (1.0 - ei))
        counts[i] += 1
        counts[j] += 1

    return ratings, counts


def ratings_dataframe(ratings: dict, counts: dict) -> pd.DataFrame:
    """Tidy ratings table sorted best -> worst."""
    df = pd.DataFrame(
        {
            "case_id": list(ratings.keys()),
            "rating": [ratings[c] for c in ratings],
            "n_compares": [counts.get(c, 0) for c in ratings],
        }
    )
    return df.sort_values("rating", ascending=False).reset_index(drop=True)


# ============================================================================== pairing ====
def draw_case(
    rng: np.random.Generator,
    case_ids,
    counts: dict,
    ratings: dict,
    exclude=(),
    near: float | None = None,
    smart: bool = True,
    spread: float = 200.0,
) -> int:
    """Pick the next case to show.

    Weighting balances two goals:
      * **coverage** -- favour cases compared fewest times so all 5,000 get sampled;
      * **information** (``smart``) -- favour cases whose current rating is close to ``near``
        (the staying champion), because near-equal match-ups are the most informative.
    """
    exclude = {int(e) for e in exclude if e is not None}
    pool = [int(c) for c in case_ids if int(c) not in exclude]
    if not pool:
        pool = [int(c) for c in case_ids]

    w = np.array([1.0 / (1.0 + counts.get(c, 0)) for c in pool], dtype=float)  # coverage
    if smart and near is not None:
        r = np.array([ratings.get(c, BASE_RATING) for c in pool], dtype=float)
        w = w * np.exp(-((r - near) ** 2) / (2.0 * spread ** 2))               # informativeness

    total = w.sum()
    p = (w / total) if total > 0 else None
    return int(rng.choice(pool, p=p))


# ============================================================================== buckets ====
def default_labels(n_buckets: int):
    if n_buckets == 3:
        return ["Low", "Moderate", "High"]
    if n_buckets == 5:
        return ["Very Low", "Low", "Moderate", "High", "Very High"]
    return [f"Tier {i + 1}" for i in range(n_buckets)]


def assign_buckets(ratings_df: pd.DataFrame, n_buckets: int = 5, labels=None) -> pd.DataFrame:
    """Split ranked cases into ``n_buckets`` equal-size capacity tiers (lowest -> highest).

    Only cases with >=1 comparison are bucketed; the rest are ``"Unranked"``. Ranking (not the
    raw rating) is quantile-cut so ties / uneven rating gaps cannot break the split.
    """
    labels = labels or default_labels(n_buckets)
    df = ratings_df.copy()
    df["bucket"] = "Unranked"
    df["bucket_idx"] = -1

    seen_mask = df["n_compares"] > 0
    seen = df.loc[seen_mask]
    n_seen = len(seen)
    if n_seen == 0:
        return df

    ranks = seen["rating"].rank(method="first")
    if n_seen >= n_buckets:
        idx = pd.qcut(ranks, n_buckets, labels=list(range(n_buckets))).astype(int)
    else:  # too few seen cases to fill every bucket -> spread them across the low tiers
        idx = ((ranks - 1) * n_buckets // n_seen).clip(0, n_buckets - 1).astype(int)

    df.loc[seen_mask, "bucket_idx"] = idx.values
    df.loc[seen_mask, "bucket"] = [labels[i] for i in idx.values]
    return df
