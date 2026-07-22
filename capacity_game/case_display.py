"""Render a clean clinical patient vignette from a ``synthetic_cases.csv`` row.

Only **osteogenic-capacity-relevant patient factors** are shown. Deliberately hidden:
  * ``capacity_true`` / ``demand_true`` / ``graft_cc_est`` -- the synthetic answer key;
  * procedure / demand fields (region, indication, approach, levels, ...) -- capacity is a
    property of the *patient*, so surgery details are omitted to keep the judgement focused.

Lab values are colour-coded by **standard clinical reference ranges** (a fast-reading aid, not a
capacity score): a patient can be red on one lab and green on another.
"""

from __future__ import annotations

import pandas as pd

# Never surface these to the surgeon.
HIDDEN = {"capacity_true", "demand_true", "graft_cc_est"}

COMORBID = {
    "osteoporosis": "Osteoporosis",
    "osteopenia": "Osteopenia",
    "diabetes": "Diabetes",
    "rheumatoid": "Rheumatoid / inflammatory arthritis",
    "renal": "Chronic kidney disease",
    "copd": "COPD",
    "chf": "Congestive heart failure",
    "hyperparathyroid": "Hyperparathyroidism",
    "cancer": "Active / prior cancer",
    "malnutrition": "Malnutrition",
}
MEDS = {
    "steroid_med": "Chronic corticosteroids",
    "immunosuppressant": "Immunosuppressant",
    "anabolic_agent": "Anabolic bone agent (teriparatide / romosozumab)",
    "bisphosphonate": "Bisphosphonate / antiresorptive",
    "nsaid": "Chronic NSAID",
    "glp1": "GLP-1 agonist",
}
HISTORY = {
    "prior_spine_surgery": "Prior spine surgery",
    "nonunion_index": "Prior non-union (index site)",
    "nonunion_other": "Prior non-union (other site)",
    "adjacent_segment": "Adjacent-segment disease",
    "radiation": "Prior spinal radiation",
    "bariatric": "Bariatric surgery",
    "transplant": "Solid-organ transplant",
}
SMOKER = {"never": "Never smoker", "former": "Former smoker", "current": "Current smoker"}


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


def active_labels(row, mapping) -> list[str]:
    return [label for col, label in mapping.items() if col in row and _flag(row[col])]


# ---- lab value -> (display text, severity level) using standard reference ranges ---------
def _bmd_level(t):
    if t is None:
        return "T-score --", "neutral"
    if t <= -2.5:
        return f"BMD T {t:+.1f} (osteoporotic)", "bad"
    if t < -1.0:
        return f"BMD T {t:+.1f} (osteopenic)", "warn"
    return f"BMD T {t:+.1f} (normal)", "good"


def _a1c_level(a):
    if a is None:
        return "A1c --", "neutral"
    if a >= 6.5:
        return f"A1c {a:.1f}% (diabetic)", "bad"
    if a >= 5.7:
        return f"A1c {a:.1f}% (pre-diabetic)", "warn"
    return f"A1c {a:.1f}% (normal)", "good"


def _vitd_level(d):
    if d is None:
        return "Vit D --", "neutral"
    if d < 20:
        return f"Vit D {d:.0f} ng/mL (deficient)", "bad"
    if d < 30:
        return f"Vit D {d:.0f} ng/mL (insufficient)", "warn"
    return f"Vit D {d:.0f} ng/mL (sufficient)", "good"


def _bmi_level(b):
    if b is None:
        return "BMI --", "neutral"
    if b < 18.5:
        return f"BMI {b:.0f} (underweight)", "warn"
    if b >= 35:
        return f"BMI {b:.0f} (obese II+)", "warn"
    return f"BMI {b:.0f}", "neutral"


def _chip(text, level):
    return f'<span class="lab-chip lab-{level}">{text}</span>'


def _badges(labels):
    if not labels:
        return '<span class="badge-none">None reported</span>'
    return "".join(f'<span class="badge">{l}</span>' for l in labels)


# --------------------------------------------------------------------------- public API ---
def short_summary(row) -> str:
    """One-line descriptor for tables / leaderboards."""
    age = _num(row.get("age"))
    bmi = _num(row.get("bmi"))
    t = _num(row.get("bmd_tscore"))
    sex = row.get("sex", "?")
    n = row.get("n_comorbid", 0)
    age_s = f"{age:.0f}" if age is not None else "?"
    bmi_s = f"{bmi:.0f}" if bmi is not None else "?"
    t_s = f"{t:+.1f}" if t is not None else "?"
    return f"{age_s}{sex}, BMI {bmi_s}, T {t_s}, {int(_num(n) or 0)} comorb."


def case_card_html(row, slot: str = "") -> str:
    """Full styled clinical card (HTML). Rendered with ``unsafe_allow_html=True``.

    ``slot`` is an optional corner tag (e.g. "A" / "B") pairing the card with its vote button.
    """
    age = _num(row.get("age"))
    bmi = _num(row.get("bmi"))
    sex = row.get("sex", "?")
    smoke = SMOKER.get(str(row.get("smoker", "")).lower(), None)

    age_s = f"{age:.0f}" if age is not None else "?"
    sex_word = {"M": "Male", "F": "Female"}.get(str(sex), str(sex))
    monogram = f"{age_s}{sex}"
    slot_tag = f"<span class='pcard-slot'>{slot}</span>" if slot else ""

    labs = "".join(
        [
            _chip(*_bmi_level(bmi)),
            _chip(*_bmd_level(_num(row.get("bmd_tscore")))),
            _chip(*_a1c_level(_num(row.get("a1c")))),
            _chip(*_vitd_level(_num(row.get("vitamin_d")))),
        ]
    )
    smoke_chip = ""
    if smoke:
        lvl = {"Never smoker": "good", "Former smoker": "warn", "Current smoker": "bad"}[smoke]
        smoke_chip = _chip(smoke, lvl)

    comorbid = _badges(active_labels(row, COMORBID))
    meds = _badges(active_labels(row, MEDS))
    hist = _badges(active_labels(row, HISTORY))

    return f"""
<div class="pcard">
  {slot_tag}
  <div class="pcard-top">
    <div class="pcard-avatar">{monogram}</div>
    <div>
      <div class="pcard-demo">{age_s}-year-old {sex_word}</div>
    </div>
  </div>
  <div class="pcard-labs">{labs}{smoke_chip}</div>
  <div class="pcard-section">
    <div class="pcard-label">Comorbidities</div>{comorbid}
  </div>
  <div class="pcard-section">
    <div class="pcard-label">Bone-relevant medications</div>{meds}
  </div>
  <div class="pcard-section">
    <div class="pcard-label">Surgical / medical history</div>{hist}
  </div>
</div>
"""
