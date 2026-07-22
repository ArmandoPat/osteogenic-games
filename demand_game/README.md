# Osteogenic Demand — Comparison Game

A fast surgeon data-collection tool. Two synthetic **planned operations** are shown side by side;
the surgeon picks the one with **higher osteogenic demand** (how much new bone the construct must
form to fuse). The winner stays on the board, the loser is swapped for a new challenger, and every
click feeds a shared **Elo** ladder that drives live pairing.

This is the demand-axis twin of the capacity game: capacity judges the *patient's* intrinsic
bone-forming ability (biology); **demand judges the *procedure*** — the surgical challenge the
construct places on healing. The card therefore shows the **surgery** (region, indication,
approach, construct span, levels, deformity, osteotomy, blood loss) and hides the patient's
biology, the mirror image of the capacity card.

Surgeons sign in from a **roster dropdown + 4-digit PIN**, so every comparison is attributed to a
stable surgeon id and each surgeon's progress is tracked independently over their lifetime. Cases
are drawn from a fixed **200-case pool** (not all 5,000) so a useful ranking is reachable, and
pairing is **personalised** — each surgeon avoids pairs they've already judged.

Pairwise "A > B" judgements are turned into a global ranking. Live pairing uses online **Elo**;
the **model-ready labels** are produced by a **surgeon-reliability-weighted Bradley-Terry** fit
(down-weighting raters who disagree with the consensus) and exposed as a 0–1 score, a z-score, and
an ordinal **tier** (Very Low → Very High). Because the cases are synthetic, the ladder is
continuously validated against the hidden `demand_true` via a Spearman correlation (an owner-only
sanity gauge — never shown during play).

## Run locally

```powershell
# from the repo root, using the project venv.
# Use a DIFFERENT port from the capacity game (8501) so both can run at once.
C:\mlenvs\bonegraft\Scripts\python.exe -m streamlit run demand_game/app.py --server.port 8502
```

A browser tab opens on a branded welcome screen. **Pick your name from the dropdown and enter your
4-digit PIN** to sign in, then start comparing. Use the mouse or keyboard: **← / →** pick the
left / right operation, **↓** (or space) skips. PINs are issued by the owner (see below); each
surgeon's comparison count and cases-seen are kept separate.

## Access: surgeons vs owner

The app has two roles:

- **Surgeon (default):** the only thing available is the comparison task and *their own* progress
  (comparisons, cases seen, next milestone) — no Insights tab, no data exports, no settings, no
  truth diagnostics, no other surgeons' data. This is what everyone sees at the plain URL.
- **Owner (you):** unlocks the **Insights** tab, model-ready exports, the **Roster & PINs**
  manager, pairing settings, and the live-vs-truth gauge.

To enter owner mode, open the app with `?owner=1` (e.g. `http://localhost:8502/?owner=1`), expand
**Owner access** in the sidebar, and enter the owner passcode. The passcode lives in
`.streamlit/secrets.toml` (gitignored, never shared):

```toml
admin_passcode = "your-strong-passcode"
```

The same `admin_passcode` unlocks both games; a per-app override is available via the
`DEMAND_ADMIN_PASSCODE` environment variable. If no passcode is configured, owner mode stays
locked (fail closed).

### Surgeon roster & PINs (owner)

Surgeons sign in from a managed roster instead of typing a free-text name. In owner mode the
sidebar shows a **Roster & PINs** panel where you can:

- **View** every surgeon's name, id (`sNNN`), 4-digit PIN, and active flag.
- **Add** a surgeon — a random PIN is generated automatically and shown once via a toast.
- **Reset PIN**, **deactivate / reactivate**, or **remove** a surgeon.

The roster is seeded with the pilot names on first run and stored in
`outputs/demand_game/roster.csv`. **That file holds plaintext PINs and is gitignored** — share
each surgeon's PIN with them privately; never commit it.

> **Security depends on hosting.** Passcode-gating only isolates surgeons from the insights when
> the app runs on a **server** and surgeons receive just the URL — the data and passcode never
> leave the server, and the compare view never sends the hidden truth (`demand_true`,
> `capacity_true`, `graft_cc_est`) to the browser. If instead you hand the code/CSV to surgeons to
> run locally, they can read `synthetic_cases.csv` (which contains the truth) directly, so
> distribute a **link, not the files**.

## Deploy for remote surgeons (optional)

Push the repo to GitHub and point [Streamlit Community Cloud](https://share.streamlit.io) at
`demand_game/app.py` (free). Surgeons then just open a link — no install. Ensure
`outputs/synthetic/synthetic_cases.csv` is committed so the app can read the cases.

## How it works

| Piece           | File                              | Notes                                                                  |
| --------------- | --------------------------------- | ---------------------------------------------------------------------- |
| Game logic      | [engine.py](engine.py)             | Elo replay (base 1500, K 32), 200-case pool, personalised pairing, self-healing vote log. |
| Batch analytics | [analysis.py](analysis.py)         | Bradley-Terry (plain + reliability-weighted), Spearman-vs-truth, inter-surgeon agreement. |
| Roster & PINs   | [roster.py](roster.py)             | Owner-managed surgeon roster with per-surgeon PIN issue / reset / active flags. |
| Model exports   | [exports.py](exports.py)           | Builds tidy `case_labels / comparisons / surgeons / cases` tables on every vote. |
| Procedure card  | [case_display.py](case_display.py) | Surgical vignette; hides the answer key + patient/capacity fields.     |
| UI              | [app.py](app.py)                   | Medtronic-branded Streamlit app. Surgeons see Compare only; owner unlocks Insights + roster. |

### Data in / out

- **Input:** `outputs/synthetic/synthetic_cases.csv` (5,000 cases). The app never shows
  `demand_true`, `capacity_true`, `graft_cc_est`, or any patient/capacity field.
- **Pool:** `outputs/demand_game/case_pool.csv` — the fixed 200-case subset actually shown
  (seeded random, stable across restarts, committable). Only votes where **both** cases are in the
  pool feed the ladder and exports, so legacy / off-pool test votes are ignored automatically.
- **Output:** `outputs/demand_game/`
  - `votes.csv` — append-only log (the source of truth). One row per click:
    `timestamp, surgeon, surgeon_id, session_id, winner_case_id, loser_case_id, pair_a_id, pair_b_id`.
    Legacy logs (pre-`surgeon_id`) are migrated to this schema automatically on load.
  - `roster.csv` — surgeon roster + PINs (**gitignored**; owner-only).
  - **Model-ready exports**, rewritten automatically on every vote and downloadable from the
    owner **Insights** tab:
    - `case_labels.csv` — one row per pool case: `demand_score`, `score_01`, `score_z`,
      ordinal `tier` / `tier_idx`, `n_compares`, joined case features, and (owner-only)
      `demand_true` as the **last** column for validation. This is the table the model trains on.
    - `comparisons.csv` — tidy fact table, one row per comparison (`comparison_id`, surgeon, winner/loser).
    - `surgeons.csv` — per-surgeon rollup (comparisons, cases seen, reliability weight, first/last
      vote). **No PINs.**
    - `cases.csv` — the pool cases with features (truth columns dropped).

The ranking is rebuilt from `votes.csv` on every load, so progress survives restarts and pools
across all surgeons.

### What "osteogenic demand" means here

Demand is a property of the **procedure**, not the patient. The synthetic `demand_true` rises with
the number of levels fused, the Cobb angle (deformity), the osteotomy grade
(none → Ponte → PSO → VCR), the spondylolisthesis grade, revision surgery, posterolateral fusion,
lordotic correction, fusion to the pelvis (S1), crossing the cervicothoracic / thoracolumbar
junctions, and estimated blood loss. The card surfaces exactly these surgical factors. The
estimated bone-graft volume (`graft_cc_est`) is **hidden** because it is a direct proxy for the
answer (target leakage).

### Game mechanic

Winner-stays "king of the hill": the chosen case is re-challenged by a new one. To keep coverage
of the 200-case pool and stay informative, challengers are drawn favouring **under-sampled** cases
and cases with a **similar current rating** (toggle "Smart pairing" off for pure random), and any
pair the **current surgeon** has already judged is skipped so each surgeon keeps seeing fresh
match-ups. After a champion wins 6 in a row it retires and a fresh pair is drawn. "Too close to
call" draws a new pair without recording a vote.

### Progress & engagement (per surgeon)

Each surgeon's **lifetime** totals are reconstructed from `votes.csv` filtered by their
`surgeon_id` — signing out and back in, or a different surgeon signing in, never mixes counts.
The sidebar shows their comparisons, cases seen (`n / 200`), and a progress bar toward the next
**milestone** (10, 25, 50, 75, 100, 150, 200, 300, 400, 500); a toast celebrates each milestone.
Surgeons are encouraged to do as many as they can — completing all cases is **not** expected — and
a gentle nudge appears after ~5 minutes of a session. Distinct "cases seen" counts only cases the
surgeon actually voted on (skips don't count).

### Notes

- `votes.csv` is append-only; ratings, labels, and exports are always derived from it
  (deterministic replay), so the log can be re-analysed at any time from a notebook.
- Ranking the 200-case pool needs many comparisons. The owner Spearman-vs-truth gauge shows how
  well the ladder is tracking; use it to decide when enough data has been collected.
- The demand game writes to its **own** `outputs/demand_game/` folder and reuses the same
  200-case pool seed as the capacity game, so a case can be labelled on both axes independently.
