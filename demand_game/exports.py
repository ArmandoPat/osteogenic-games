"""Model-ready export pipeline: turn the append-only vote log into tidy, normalised, joinable
tables the demand model can train on. Regenerated on every vote (cheap for the reduced pool).

Tables written to ``outputs/demand_game/``:
  * ``comparisons.csv`` -- tidy fact table (one row per pairwise judgement);
  * ``surgeons.csv``    -- one row per rater (NO PINs): contribution + reliability weight;
  * ``cases.csv``       -- pool cases with their procedure features (dimension table);
  * ``case_labels.csv`` -- MODEL-READY: per case a surgeon-normalised demand score
    (continuous + 0-1 + z), an ordinal tier, ``n_compares``, joined features, and
    (owner-only, last column) the hidden ``demand_true`` for validation.

Normalisation = pool everyone onto ONE shared ladder via a Bradley-Terry fit whose comparisons
are weighted by each surgeon's reliability (agreement with consensus), so noisy raters count
less, instead of assuming every surgeon is equally scaled/strict.

Pure Python (no Streamlit).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import analysis
import engine

TRUTH_COLS = ("capacity_true", "demand_true", "graft_cc_est")
N_TIERS = 5
TIER_LABELS = ["Very Low", "Low", "Moderate", "High", "Very High"]


def _scale(s: pd.Series):
    """Return (min-max 0-1, z-score) of a numeric series, robust to zero spread."""
    x = s.astype(float)
    span = x.max() - x.min()
    s01 = (x - x.min()) / span if span > 0 else pd.Series(0.5, index=x.index)
    sd = x.std(ddof=0)
    sz = (x - x.mean()) / sd if sd > 0 else pd.Series(0.0, index=x.index)
    return s01, sz


def build_case_labels(votes, cases, pool_ids, ratings_df=None, weights=None) -> pd.DataFrame:
    """The model-ready per-case label table (surgeon-normalised score + tier + features)."""
    pool = [int(c) for c in pool_ids]
    if ratings_df is None:
        ratings, counts = engine.elo_replay(votes, pool)
        ratings_df = engine.ratings_dataframe(ratings, counts)
    df = ratings_df[ratings_df["case_id"].isin(pool)].copy()

    if weights is None:
        weights = analysis.surgeon_reliability(votes)
    bt = analysis.weighted_bradley_terry(votes, pool, weights=weights) if len(votes) else None
    if bt is not None:
        df = df.merge(bt, on="case_id", how="left")
    else:
        df["bt_strength"] = np.nan

    # Continuous surgeon-normalised score: pooled BT strength where available, else the Elo rating.
    df["demand_score"] = df["bt_strength"].where(df["bt_strength"].notna(), df["rating"])

    seen = df["n_compares"] > 0
    df["score_01"] = np.nan
    df["score_z"] = np.nan
    df["tier"] = "Unranked"
    df["tier_idx"] = -1
    if seen.any():
        s01, sz = _scale(df.loc[seen, "demand_score"])
        df.loc[seen, "score_01"] = s01
        df.loc[seen, "score_z"] = sz
        ranks = df.loc[seen, "demand_score"].rank(method="first")
        n = int(seen.sum())
        if n >= N_TIERS:
            idx = pd.qcut(ranks, N_TIERS, labels=list(range(N_TIERS))).astype(int)
        else:
            idx = ((ranks - 1) * N_TIERS // n).clip(0, N_TIERS - 1).astype(int)
        df.loc[seen, "tier_idx"] = idx.values
        df.loc[seen, "tier"] = [TIER_LABELS[i] for i in idx.values]

    feat_cols = [c for c in cases.columns if c not in TRUTH_COLS and c != "case_id"]
    keep = ["case_id", *feat_cols] + (["demand_true"] if "demand_true" in cases.columns else [])
    df = df.merge(cases[keep], on="case_id", how="left")

    lead = ["case_id", "demand_score", "score_01", "score_z", "tier", "tier_idx",
            "rating", "bt_strength", "n_compares"]
    has_truth = "demand_true" in df.columns
    tail = [c for c in df.columns if c not in lead and c != "demand_true"]
    ordered = [*lead, *tail] + (["demand_true"] if has_truth else [])  # truth LAST = owner-only
    df = df[ordered]
    return df.sort_values("demand_score", ascending=False).reset_index(drop=True)


def build_surgeons(votes, roster_df=None, weights=None) -> pd.DataFrame:
    """One row per rater (no PINs): comparisons made, distinct cases seen, reliability weight."""
    cols = ["surgeon_id", "display_name", "n_comparisons", "n_cases_seen",
            "reliability", "first_vote", "last_vote"]
    if votes is None or len(votes) == 0:
        return pd.DataFrame(columns=cols)
    v = votes.copy()
    if "surgeon_id" not in v.columns:
        v["surgeon_id"] = pd.NA
    if "surgeon" not in v.columns:
        v["surgeon"] = pd.NA
    weights = weights if weights is not None else analysis.surgeon_reliability(v)

    key = v["surgeon_id"].astype("object").where(v["surgeon_id"].notna(), v["surgeon"])
    rows = []
    for sid, g in v.groupby(key):
        disp = g["surgeon"].dropna().iloc[0] if g["surgeon"].notna().any() else str(sid)
        seen_cols = [g["winner_case_id"], g["loser_case_id"]]
        if "outcome" in g.columns:  # tie rows carry their pair in pair_a/pair_b, not winner/loser
            tg = g[g["outcome"].astype(str).str.lower() == "tie"]
            seen_cols += [tg["pair_a_id"], tg["pair_b_id"]]
        seen = pd.unique(pd.concat(seen_cols).dropna())
        rows.append({
            "surgeon_id": sid,
            "display_name": disp,
            "n_comparisons": int(len(g)),
            "n_cases_seen": int(len(seen)),
            "reliability": round(float(weights.get(disp, 1.0)), 3),
            "first_vote": g["timestamp"].min() if "timestamp" in g else pd.NA,
            "last_vote": g["timestamp"].max() if "timestamp" in g else pd.NA,
        })
    out = pd.DataFrame(rows, columns=cols)
    if roster_df is not None and len(roster_df):
        name_map = dict(zip(roster_df["surgeon_id"], roster_df["display_name"]))
        out["display_name"] = out["surgeon_id"].map(name_map).fillna(out["display_name"])
    return out.sort_values("n_comparisons", ascending=False).reset_index(drop=True)


def build_comparisons(votes) -> pd.DataFrame:
    """Tidy fact table: one row per judgement, with a stable ``comparison_id``."""
    cols = ["timestamp", "surgeon_id", "surgeon", "session_id", "outcome",
            "winner_case_id", "loser_case_id", "pair_a_id", "pair_b_id"]
    if votes is None or len(votes) == 0:
        return pd.DataFrame(columns=["comparison_id", *cols])
    v = votes.copy()
    for c in cols:
        if c not in v.columns:
            v[c] = pd.NA
    v = v[cols].reset_index(drop=True)
    v.insert(0, "comparison_id", range(1, len(v) + 1))
    return v


def build_cases_dim(cases, pool_ids) -> pd.DataFrame:
    """Pool cases with procedure-feature columns (truth columns dropped)."""
    pool = [int(c) for c in pool_ids]
    feat_cols = [c for c in cases.columns if c not in TRUTH_COLS]
    return cases[cases["case_id"].isin(pool)][feat_cols].reset_index(drop=True)


def regenerate(out_dir, votes, cases, pool_ids, roster_df=None) -> pd.DataFrame:
    """Recompute and write all tidy / model-ready tables. Returns the ``case_labels`` frame."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pv = engine.filter_pool_votes(votes, pool_ids)
    weights = analysis.surgeon_reliability(pv)

    labels = build_case_labels(pv, cases, pool_ids, weights=weights)
    surgeons = build_surgeons(pv, roster_df=roster_df, weights=weights)
    comparisons = build_comparisons(pv)
    cases_dim = build_cases_dim(cases, pool_ids)

    labels.to_csv(out_dir / "case_labels.csv", index=False)
    surgeons.to_csv(out_dir / "surgeons.csv", index=False)
    comparisons.to_csv(out_dir / "comparisons.csv", index=False)
    cases_dim.to_csv(out_dir / "cases.csv", index=False)
    return labels
