"""Owner-managed surgeon roster with per-surgeon 4-digit PINs.

The roster is a small CSV the **owner** controls -- both by editing the file directly and from
the app's owner mode. It is **git-ignored** because it holds sign-in PINs. Surgeons pick their
name from a dropdown and enter their PIN; the app never exposes anyone else's PIN to a surgeon.

Pure Python (no Streamlit) so it can be unit-tested and driven from notebooks.
"""

from __future__ import annotations

import hmac
import secrets
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:  # optional durable SQL backend for ephemeral cloud hosts (see storage.py)
    import storage
except Exception:  # pragma: no cover - storage backend is optional
    storage = None

ROSTER_COLS = ["surgeon_id", "display_name", "pin", "active", "created"]

# Seeded on first run so the app is usable out of the box; the owner can edit / reset afterwards.
SEED_NAMES = [
    "Dr. Crawford", "Dr. Berven", "Dr. Patel", "Dr. Mesfin",
    "Dr. Perry", "Dr. Whang", "Dr. Panchal", "Dr. Van",
]


def roster_path(out_dir) -> Path:
    return Path(out_dir) / "roster.csv"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gen_pin() -> str:
    """A cryptographically-random 4-digit PIN (leading zeros kept)."""
    return f"{secrets.randbelow(10000):04d}"


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=ROSTER_COLS)


def load_roster(path) -> pd.DataFrame:
    """Read the roster, returning an empty typed frame if it does not exist. PINs stay strings
    (so leading zeros survive) and ``active`` is coerced to int. When a durable SQL backend is
    configured (deployment) the roster is read from it instead."""
    if storage is not None and storage.enabled():
        df = storage.load_table("roster", ROSTER_COLS)
        if len(df) == 0:
            return _empty()
        df["pin"] = df["pin"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
        df["active"] = pd.to_numeric(df["active"], errors="coerce").fillna(1).astype(int)
        return df[ROSTER_COLS].reset_index(drop=True)
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        df = pd.read_csv(path, dtype={"pin": str})
        for c in ROSTER_COLS:
            if c not in df.columns:
                df[c] = pd.NA
        df["pin"] = df["pin"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
        df["active"] = pd.to_numeric(df["active"], errors="coerce").fillna(1).astype(int)
        return df[ROSTER_COLS].reset_index(drop=True)
    return _empty()


def save_roster(path, df: pd.DataFrame) -> None:
    if storage is not None and storage.enabled():
        storage.replace_table("roster", df, ROSTER_COLS)
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[ROSTER_COLS].to_csv(path, index=False)


def next_id(df: pd.DataFrame) -> str:
    """Next sequential ``sNNN`` id (stable, opaque, rename-safe)."""
    n = 0
    for sid in df.get("surgeon_id", []):
        s = str(sid)
        if s.startswith("s") and s[1:].isdigit():
            n = max(n, int(s[1:]))
    return f"s{n + 1:03d}"


def seed_if_empty(path, names=None) -> pd.DataFrame:
    """Create a starter roster (one auto-PIN per name) only when no roster exists yet."""
    df = load_roster(path)
    if len(df) > 0:
        return df
    names = names or SEED_NAMES
    rows = [
        {"surgeon_id": f"s{i:03d}", "display_name": name, "pin": gen_pin(),
         "active": 1, "created": _now()}
        for i, name in enumerate(names, start=1)
    ]
    df = pd.DataFrame(rows, columns=ROSTER_COLS)
    save_roster(path, df)
    return df


def add_surgeon(path, display_name: str) -> tuple:
    """Append a surgeon with a fresh auto-generated PIN. Returns ``(roster, new_row)``."""
    df = load_roster(path)
    row = {"surgeon_id": next_id(df), "display_name": display_name.strip(),
           "pin": gen_pin(), "active": 1, "created": _now()}
    df = pd.concat([df, pd.DataFrame([row], columns=ROSTER_COLS)], ignore_index=True)
    save_roster(path, df)
    return df, row


def reset_pin(path, surgeon_id: str) -> str:
    df = load_roster(path)
    pin = gen_pin()
    df.loc[df["surgeon_id"] == surgeon_id, "pin"] = pin
    save_roster(path, df)
    return pin


def set_active(path, surgeon_id: str, active: bool) -> None:
    df = load_roster(path)
    df.loc[df["surgeon_id"] == surgeon_id, "active"] = int(bool(active))
    save_roster(path, df)


def remove_surgeon(path, surgeon_id: str) -> None:
    """Drop a surgeon from the roster. Their past votes remain in ``votes.csv``."""
    df = load_roster(path)
    save_roster(path, df[df["surgeon_id"] != surgeon_id])


def active_surgeons(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df
    return df[df["active"].astype(int) == 1].reset_index(drop=True)


def verify_pin(df: pd.DataFrame, surgeon_id: str, pin: str) -> bool:
    """Constant-time PIN check for a given surgeon id."""
    row = df[df["surgeon_id"] == surgeon_id]
    if len(row) == 0:
        return False
    real = str(row.iloc[0]["pin"]).zfill(4)
    return hmac.compare_digest(str(pin).strip().zfill(4), real)
