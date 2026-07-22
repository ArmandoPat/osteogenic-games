"""Batch analytics for the comparison game: a principled Bradley-Terry cross-check of the
online Elo ladder, validation against the hidden synthetic truth, and inter-surgeon agreement.
Pure Python (no Streamlit)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr


def bradley_terry(votes: pd.DataFrame, case_ids, C: float = 1.0) -> pd.DataFrame | None:
    """L2-regularised Bradley-Terry latent strengths via logistic regression.

    Each comparison contributes a design row ``e_winner - e_loser`` with label 1; the mirror
    row ``e_loser - e_winner`` with label 0 supplies the second class. The (ridge) penalty keeps
    the estimate stable even when the comparison graph is sparse or disconnected -- unlike a raw
    MLE Bradley-Terry fit. Coefficients are the latent demand strengths (higher = more demand).

    Only cases that actually appear in the votes are estimated (others have no signal).
    """
    from sklearn.linear_model import LogisticRegression

    if votes is None or len(votes) == 0:
        return None
    v = votes.dropna(subset=["winner_case_id", "loser_case_id"])
    if len(v) == 0:
        return None

    seen = pd.unique(pd.concat([v["winner_case_id"], v["loser_case_id"]]).astype(int))
    seen = [int(c) for c in seen]
    idx = {c: i for i, c in enumerate(seen)}
    n_items = len(seen)

    rows, cols, data = [], [], []
    for r, (w, l) in enumerate(zip(v["winner_case_id"].astype(int), v["loser_case_id"].astype(int))):
        rows += [r, r]
        cols += [idx[w], idx[l]]
        data += [1.0, -1.0]
    X = sparse.csr_matrix((data, (rows, cols)), shape=(len(v), n_items))

    # mirror to create both logistic classes
    X2 = sparse.vstack([X, -X], format="csr")
    y2 = np.concatenate([np.ones(X.shape[0]), np.zeros(X.shape[0])])

    clf = LogisticRegression(fit_intercept=False, C=C, max_iter=2000)
    clf.fit(X2, y2)
    strengths = clf.coef_.ravel()

    out = pd.DataFrame({"case_id": seen, "bt_strength": strengths})
    return out.sort_values("bt_strength", ascending=False).reset_index(drop=True)


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
    """Bradley-Terry latent strengths with per-comparison **sample weights** (surgeon
    reliability), pooling everyone onto ONE shared, surgeon-normalised ladder.

    Identical design to :func:`bradley_terry` but each comparison row (and its mirror) carries
    its rater's weight, so low-agreement surgeons pull the estimate less. Falls back to unit
    weights when ``weights`` is None or the surgeon column is absent.
    """
    from sklearn.linear_model import LogisticRegression

    if votes is None or len(votes) == 0:
        return None
    v = votes.dropna(subset=["winner_case_id", "loser_case_id"])
    if len(v) == 0:
        return None

    seen = pd.unique(pd.concat([v["winner_case_id"], v["loser_case_id"]]).astype(int))
    seen = [int(c) for c in seen]
    idx = {c: i for i, c in enumerate(seen)}

    rows, cols, data = [], [], []
    for r, (w, l) in enumerate(zip(v["winner_case_id"].astype(int), v["loser_case_id"].astype(int))):
        rows += [r, r]
        cols += [idx[w], idx[l]]
        data += [1.0, -1.0]
    X = sparse.csr_matrix((data, (rows, cols)), shape=(len(v), len(seen)))

    if weights is not None and "surgeon" in v.columns:
        sw = v["surgeon"].map(lambda s: weights.get(s, 1.0)).to_numpy(dtype=float)
    else:
        sw = np.ones(len(v))

    X2 = sparse.vstack([X, -X], format="csr")
    y2 = np.concatenate([np.ones(X.shape[0]), np.zeros(X.shape[0])])
    sw2 = np.concatenate([sw, sw])

    clf = LogisticRegression(fit_intercept=False, C=C, max_iter=2000)
    clf.fit(X2, y2, sample_weight=sw2)
    out = pd.DataFrame({"case_id": seen, "bt_strength": clf.coef_.ravel()})
    return out.sort_values("bt_strength", ascending=False).reset_index(drop=True)
