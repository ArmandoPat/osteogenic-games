"""Render a clean surgical/procedure vignette from a ``synthetic_cases.csv`` row.

Only **osteogenic-demand-relevant procedure factors** are shown. Deliberately hidden:
  * ``capacity_true`` / ``demand_true`` / ``graft_cc_est`` -- the synthetic answer key
    (``graft_cc_est`` is a direct proxy for demand, so it is target leakage);
  * patient / capacity fields (age, labs, comorbidities, medications, history) -- demand is a
    property of the *procedure*, so patient biology is omitted to keep the judgement focused.

Procedure values are colour-coded by **surgical complexity** (a fast-reading aid, not a demand
score): a case can be red on one factor (e.g. a VCR osteotomy) and green on another (single level).
"""

from __future__ import annotations

import pandas as pd

# Never surface these to the surgeon (synthetic answer key + demand proxy).
HIDDEN = {"capacity_true", "demand_true", "graft_cc_est"}

# Vertebral ordering (C2 -> S1) used to detect high-demand junction crossings / fusion to pelvis.
_SPINE_ORDER = (["C2", "C3", "C4", "C5", "C6", "C7"]
                + [f"T{i}" for i in range(1, 13)]
                + [f"L{i}" for i in range(1, 6)] + ["S1"])
_LVL = {lvl: i for i, lvl in enumerate(_SPINE_ORDER)}
_T1, _T12, _L1 = _LVL["T1"], _LVL["T12"], _LVL["L1"]

APPROACH_FULL = {
    "ACDF": "ACDF (anterior cervical)",
    "ACCF": "ACCF (anterior cervical corpectomy)",
    "Posterior Cervical": "Posterior cervical",
    "PSF": "PSF (posterior spinal fusion)",
    "TLIF": "TLIF (transforaminal interbody)",
    "PLIF": "PLIF (posterior interbody)",
    "ALIF": "ALIF (anterior interbody)",
    "LLIF": "LLIF (lateral interbody)",
}


# ------------------------------------------------------------------------------- helpers --
def _flag(v) -> bool:
    try:
        return int(float(v)) == 1
    except (ValueError, TypeError):
        return False


def _num(v):
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (ValueError, TypeError):
        return None


def _chip(text, level):
    return f'<span class="lab-chip lab-{level}">{text}</span>'


def _badges(labels):
    if not labels:
        return '<span class="badge-none">None reported</span>'
    return "".join(f'<span class="badge">{l}</span>' for l in labels)


# ---- procedure factor -> (display text, severity level) by surgical complexity ------------
def _levels_level(n):
    if n is None:
        return "-- levels", "neutral"
    n = int(n)
    unit = "level" if n == 1 else "levels"
    if n <= 2:
        return f"{n} {unit}", "good"
    if n <= 4:
        return f"{n} {unit}", "warn"
    return f"{n} {unit}", "bad"


def _osteotomy_level(o):
    o = str(o or "").strip()
    if o in ("", "No osteotomy", "None", "nan"):
        return "No osteotomy", "good"
    if o == "Ponte":
        return "Ponte osteotomy", "warn"
    return f"{o} osteotomy", "bad"   # PSO / VCR (three-column) = highest demand


def _cobb_level(c):
    if c is None:
        return "Cobb --", "neutral"
    if c < 20:
        return f"Cobb {c:.0f}\u00b0", "good"
    if c < 40:
        return f"Cobb {c:.0f}\u00b0 (moderate)", "warn"
    return f"Cobb {c:.0f}\u00b0 (severe)", "bad"


def _ebl_level(e):
    if e is None:
        return "EBL --", "neutral"
    if e < 300:
        return f"EBL {e:.0f} mL", "good"
    if e < 800:
        return f"EBL {e:.0f} mL", "warn"
    return f"EBL {e:.0f} mL (high)", "bad"


# --------------------------------------------------------------------------- public API ---
def short_summary(row) -> str:
    """One-line descriptor for tables / leaderboards."""
    n = _num(row.get("n_levels"))
    cobb = _num(row.get("cobb_angle"))
    approach = row.get("approach", "?")
    indication = row.get("indication", "?")
    osteo = str(row.get("osteotomy", "") or "").strip()
    osteo_s = "no ost." if osteo in ("", "No osteotomy", "None", "nan") else osteo
    n_s = f"{int(n)}" if n is not None else "?"
    cobb_s = f"{cobb:.0f}\u00b0" if cobb is not None else "?"
    return f"{n_s}L {approach}, {indication}, Cobb {cobb_s}, {osteo_s}"


def case_card_html(row, slot: str = "") -> str:
    """Full styled surgical card (HTML). Rendered with ``unsafe_allow_html=True``.

    ``slot`` is an optional corner tag (e.g. "A" / "B") pairing the card with its vote button.
    """
    n_levels = int(_num(row.get("n_levels")) or 0)
    approach = str(row.get("approach", "?"))
    indication = str(row.get("indication", "?"))
    region = str(row.get("region", "?"))
    start = str(row.get("construct_start", "?"))
    end = str(row.get("construct_end", "?"))

    monogram = f"{n_levels}L" if n_levels else "--"
    approach_full = APPROACH_FULL.get(approach, approach)
    lvl_word = "level" if n_levels == 1 else "levels"
    slot_tag = f"<span class='pcard-slot'>{slot}</span>" if slot else ""

    labs = "".join(
        [
            _chip(*_levels_level(n_levels)),
            _chip(*_osteotomy_level(row.get("osteotomy"))),
            _chip(*_cobb_level(_num(row.get("cobb_angle")))),
            _chip(*_ebl_level(_num(row.get("ebl_ml")))),
        ]
    )

    # ---- Section 1: construct span (region, approach, levels, junction crossings) ----
    construct = [f"Region: {region}", f"Approach: {approach_full}", f"Fusion: {start} \u2192 {end}"]
    si, ei = _LVL.get(start), _LVL.get(end)
    if end == "S1":
        construct.append("Fusion to pelvis (S1)")
    if si is not None and ei is not None:
        if si < _T1 <= ei:
            construct.append("Crosses cervicothoracic junction")
        if si <= _T12 and ei >= _L1:
            construct.append("Crosses thoracolumbar junction")

    # ---- Section 2: deformity & correction ----
    deformity = []
    spondy = int(_num(row.get("spondy_grade")) or 0)
    if spondy > 0:
        deformity.append(f"Spondylolisthesis grade {spondy}")
    lordo = _num(row.get("lordotic_correction"))
    if lordo is not None and lordo >= 5:
        deformity.append(f"Lordotic correction {lordo:.0f}\u00b0")
    osteo = str(row.get("osteotomy", "") or "").strip()
    if osteo not in ("", "No osteotomy", "None", "nan"):
        deformity.append(f"{osteo} osteotomy")

    # ---- Section 3: procedure burden ----
    burden = [f"Indication: {indication}"]
    if indication == "Revision":
        burden.append("Revision procedure")
    if _flag(row.get("posterolateral")):
        burden.append("Posterolateral fusion")

    return f"""
<div class="pcard">
  {slot_tag}
  <div class="pcard-top">
    <div class="pcard-avatar">{monogram}</div>
    <div>
      <div class="pcard-demo">{n_levels}-{lvl_word} {approach}</div>
      <div class="pcard-sub">{indication} &middot; {region}</div>
    </div>
  </div>
  <div class="pcard-labs">{labs}</div>
  <div class="pcard-section">
    <div class="pcard-label">Construct</div>{_badges(construct)}
  </div>
  <div class="pcard-section">
    <div class="pcard-label">Deformity &amp; correction</div>{_badges(deformity)}
  </div>
  <div class="pcard-section">
    <div class="pcard-label">Procedure burden</div>{_badges(burden)}
  </div>
</div>
"""
