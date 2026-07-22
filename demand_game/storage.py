"""Optional durable storage backend for deployments with an EPHEMERAL filesystem.

On a normal server (or your laptop) the app keeps votes and the roster in CSV files, which is
perfect because the disk persists. On a free cloud host such as Streamlit Community Cloud the
container's disk is wiped on every restart / redeploy, so those CSVs -- and every surgeon's
collected vote -- would be lost.

When the environment variable ``BONEGRAFT_DB_URL`` is set, this module stores the votes and the
roster in a SQL database instead, so they survive restarts. When it is NOT set, the app behaves
exactly as before (local CSV): nothing changes for local runs or a persistent-disk server.

``BONEGRAFT_TABLE_PREFIX`` (e.g. ``capacity_`` / ``demand_``) lets the two games safely share a
single database without colliding.

Pure Python (no Streamlit) so engine.py / roster.py stay unit-testable.
"""

from __future__ import annotations

import os
import threading

import pandas as pd

# Serialises writes within this single process (Streamlit runs one process per app).
_WRITE_LOCK = threading.Lock()
_ENGINE = None  # lazily-created SQLAlchemy engine, cached for the life of the process


def _url():
    """The configured database URL, or None when running in local-CSV mode."""
    url = os.environ.get("BONEGRAFT_DB_URL")
    url = url.strip() if url else ""
    if not url:
        return None
    # Accept the common "postgres://" copy/paste form that SQLAlchemy no longer recognises.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def enabled() -> bool:
    """True when a database URL is configured (deployment mode); False -> use local CSV."""
    return _url() is not None


def _prefix() -> str:
    return os.environ.get("BONEGRAFT_TABLE_PREFIX", "").strip()


def _table(name: str) -> str:
    return f"{_prefix()}{name}"


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        from sqlalchemy import create_engine
        # pool_pre_ping recycles connections dropped by the DB/pooler between page loads.
        _ENGINE = create_engine(_url(), pool_pre_ping=True)
    return _ENGINE


def load_table(name: str, columns) -> pd.DataFrame:
    """Read a whole table into a DataFrame with exactly ``columns`` (missing filled NA, extras
    dropped). Returns an empty typed frame if the table does not exist yet."""
    columns = list(columns)
    try:
        from sqlalchemy import text
        eng = _get_engine()
        with eng.connect() as cx:
            df = pd.read_sql(text(f'SELECT * FROM "{_table(name)}"'), cx)
    except Exception:
        return pd.DataFrame(columns=columns)
    for c in columns:
        if c not in df.columns:
            df[c] = pd.NA
    return df[columns]


def append_row(name: str, row: dict, columns) -> None:
    """Append one row, creating the table on first write."""
    columns = list(columns)
    frame = pd.DataFrame([{c: row.get(c) for c in columns}], columns=columns)
    with _WRITE_LOCK:
        frame.to_sql(_table(name), _get_engine(), if_exists="append", index=False)


def replace_table(name: str, df: pd.DataFrame, columns) -> None:
    """Overwrite an entire table with ``df`` -- used for the small, rarely-changed roster and
    for the one-time history import."""
    columns = list(columns)
    frame = df.reindex(columns=columns)
    with _WRITE_LOCK:
        frame.to_sql(_table(name), _get_engine(), if_exists="replace", index=False)


def table_is_empty(name: str, columns) -> bool:
    """True when the table is missing or holds no rows (used to avoid clobbering live data)."""
    return len(load_table(name, columns)) == 0
