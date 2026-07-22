"""Batch analytics for the comparison game: a principled Bradley-Terry cross-check of the
online Elo ladder, validation against the hidden synthetic truth, and inter-surgeon agreement.
Pure Python (no Streamlit)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr


def _bt_comparisons(votes, weights=None):
    """Flatten a vote log into weighted (winner_case, loser_case, weight) comparison rows.

    A **pick** is one row (winner beats loser) at the rater's weight. A **tie** ("too close to
    call") is two half-weight rows -- each case "beats" the other -- which pulls the two latent
    strengths together, the Bradley-Terry analogue of a draw.
    """
    v = votes.copy()
    has_outcome = "outcome" in v.columns
    is_tie = ((v["outcome"].astype(str).str.lower() == "tie")
              if has_outcome else pd.Series(False, index=v.index))

    def _w(surg):
        return float(weights.get(surg, 1.0)) if weights is not None else 1.0

    picks = v[~is_tie].dropna(subset=["winner_case_id", "loser_case_id"])
    ties = v[is_tie].dropna(subset=["pair_a_id", "pair_b_id"]) if has_outcome else v.iloc[0:0]

    comps = []  # (winner_case, loser_case, weight)
    p_surg = picks["surgeon"] if "surgeon" in picks.columns else pd.Series(index=picks.index, dtype=object)
    for surg, w, l in zip(p_surg, picks["winner_case_id"].astype(int), picks["loser_case_id"].astype(int)):
        comps.append((int(w), int(l), _w(surg)))
    t_surg = ties["surgeon"] if "surgeon" in ties.columns else pd.Series(index=ties.index, dtype=object)
    for surg, a, b in zip(t_surg, ties["pair_a_id"].astype(int), ties["pair_b_id"].astype(int)):
        half = 0.5 * _w(surg)
        comps.append((int(a), int(b), half))
        comps.append((int(b), int(a), half))
    return comps


def _bt_fit(votes: pd.DataFrame, weights: dict | None = None, C: float = 1.0) -> pd.DataFrame | None:
    """L2-regularised, draw-aware Bradley-Terry latent strengths via weighted logistic regression.

    Each comparison contributes a design row ``e_winner - e_loser`` (label 1) plus its mirror
    (label 0) so both logistic classes are present; per-row sample weights carry surgeon
    reliability (and the 0.5 split for ties). The ridge penalty keeps the estimate stable on a
    sparse / disconnected comparison graph. Only cases that actually appear are estimated.
    """
    from sklearn.linear_model import LogisticRegression

    if votes is None or len(votes) == 0:
        return None
    comps = _bt_comparisons(votes, weights)
    if not comps:
        return None

    seen = sorted({c for (w, l, _) in comps for c in (w, l)})
    idx = {c: i for i, c in enumerate(seen)}

    rows, cols, data = [], [], []
    for r, (w, l, _wt) in enumerate(comps):
        rows += [r, r]
        cols += [idx[w], idx[l]]
        data += [1.0, -1.0]
    X = sparse.csr_matrix((data, (rows, cols)), shape=(len(comps), len(seen)))
    sw = np.array([wt for (_, _, wt) in comps], dtype=float)

    X2 = sparse.vstack([X, -X], format="csr")
    y2 = np.concatenate([np.ones(len(comps)), np.zeros(len(comps))])
    sw2 = np.concatenate([sw, sw])

    clf = LogisticRegression(fit_intercept=False, C=C, max_iter=2000)
    clf.fit(X2, y2, sample_weight=sw2)
    out = pd.DataFrame({"case_id": seen, "bt_strength": clf.coef_.ravel()})
    return out.sort_values("bt_strength", ascending=False).reset_index(drop=True)


def bradley_terry(votes: pd.DataFrame, case_ids, C: float = 1.0) -> pd.DataFrame | None:
    """Unweighted draw-aware Bradley-Terry latent strengths (higher = more demand)."""
    return _bt_fit(votes, weights=None, C=C)


def spearman_vs_truth(ratings_df: pd.DataFrame, cases_df: pd.DataFrame,
                      score_col: str = "rating", truth_col: str = "demand_true"):
    """Spearman rank correlation between a score column and the hidden synthetic truth.

    Returns ``(rho, n)`` over the compared cases, or ``None`` if too few to correlate.
    """
    if truth_col not in cases_df.columns:
        return None
    m = ratings_df.merge(cases_df[["case_id", truth_col]], on="case_id", how="left")
    if "n_compares" in m.columns:
        m = m[m["n_compares"] > 0]
    m = m.dropna(subset=[score_col, truth_col])
    if len(m) < 3:
        return None
    rho = spearmanr(m[score_col], m[truth_col]).statistic
    return float(rho), int(len(m))


def surgeon_summary(votes: pd.DataFrame) -> pd.DataFrame:
    """Per-surgeon vote counts (contribution to the shared ladder)."""
    if votes is None or len(votes) == 0:
        return pd.DataFrame(columns=["surgeon", "votes"])
    return (
        votes.groupby("surgeon", dropna=False)
        .size()
        .reset_index(name="votes")
        .sort_values("votes", ascending=False)
        .reset_index(drop=True)
    )


def inter_surgeon_agreement(votes: pd.DataFrame):
    """Fraction of repeat-judged unordered pairs where surgeons agree on the direction.

    For every unordered case pair judged by 2+ *different* surgeons, compare each surgeon's
    majority winner. Returns ``(agreement_rate, n_shared_pairs)`` or ``None`` if none overlap.
    A reliability signal analogous to inter-rater agreement.
    """
    if votes is None or len(votes) == 0:
        return None
    v = votes.dropna(subset=["winner_case_id", "loser_case_id", "surgeon"]).copy()
    if len(v) == 0:
        return None
    v["lo"] = v[["winner_case_id", "loser_case_id"]].min(axis=1).astype(int)
    v["hi"] = v[["winner_case_id", "loser_case_id"]].max(axis=1).astype(int)

    agree = total = 0
    for (lo, hi), grp in v.groupby(["lo", "hi"]):
        # each surgeon's majority pick for this pair
        picks = {}
        for surg, sg in grp.groupby("surgeon"):
            picks[surg] = int(sg["winner_case_id"].astype(int).mode().iloc[0])
        if len(picks) < 2:
            continue
        total += 1
        if len(set(picks.values())) == 1:
            agree += 1
    if total == 0:
        return None
    return agree / total, total


def surgeon_reliability(votes: pd.DataFrame, floor: float = 0.25) -> dict:
    """Per-surgeon reliability weight in ``[floor, 1]`` from agreement with the *consensus* of
    the other surgeons on pairs judged by 2+ raters.

    This is how surgeons are **normalised before pooling**: rather than assuming every rater is
    on the same scale, noisy / low-agreement raters are down-weighted when everyone is combined
    onto one shared Bradley-Terry ladder. Surgeons who never overlap anyone get a neutral 1.0.
    Returns ``{surgeon_display_name: weight}``.
    """
    if votes is None or len(votes) == 0:
        return {}
    v = votes.dropna(subset=["winner_case_id", "loser_case_id", "surgeon"]).copy()
    if len(v) == 0:
        return {}
    v["lo"] = v[["winner_case_id", "loser_case_id"]].min(axis=1).astype(int)
    v["hi"] = v[["winner_case_id", "loser_case_id"]].max(axis=1).astype(int)

    tally: dict = {}  # surgeon -> [n_agree, n_total]
    for _, grp in v.groupby(["lo", "hi"]):
        picks = {surg: int(sg["winner_case_id"].astype(int).mode().iloc[0])
                 for surg, sg in grp.groupby("surgeon")}
        if len(picks) < 2:
            continue
        for surg, pick in picks.items():
            others = [p for s, p in picks.items() if s != surg]
            consensus = max(set(others), key=others.count)
            a, t = tally.get(surg, (0, 0))
            tally[surg] = (a + int(pick == consensus), t + 1)

    weights = {surg: (floor + (1.0 - floor) * (a / t) if t else 1.0)
               for surg, (a, t) in tally.items()}
    for surg in v["surgeon"].unique():   # non-overlapping raters -> neutral
        weights.setdefault(surg, 1.0)
    return weights


def weighted_bradley_terry(votes: pd.DataFrame, case_ids, weights: dict | None = None,
                           C: float = 1.0) -> pd.DataFrame | None:
    """Draw-aware Bradley-Terry with per-comparison **sample weights** (surgeon reliability),
    pooling everyone onto ONE shared, surgeon-normalised ladder. Ties are split into two
    half-weight comparisons. Falls back to unit weights when ``weights`` is None.
    """
    return _bt_fit(votes, weights=weights, C=C)
