"""DBOS initialization for exoclaw.

Call init_dbos() once at app startup, after set_turn_context().
DBOS.launch() automatically recovers any incomplete workflows.
"""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger()


def _patch_dbos_sqlite_migrations() -> None:
    """Fix DBOS bug: unixepoch('subsec') requires SQLite >= 3.42.0 but
    DBOS checks Python version instead.

    The migration SQL uses f-strings evaluated at import time, so we
    must patch the already-evaluated migration strings, not the function.
    """
    import sqlite3

    if sqlite3.sqlite_version_info >= (3, 42, 0):
        return

    try:
        import dbos._migration as m

        bad = "(unixepoch('subsec') * 1000)"
        good = "(strftime('%s','now') * 1000)"

        # Patch pre-evaluated migration string constants
        for attr in dir(m):
            if attr.startswith("sqlite_migration_"):
                val = getattr(m, attr)
                if isinstance(val, str) and bad in val:
                    setattr(m, attr, val.replace(bad, good))

        # Patch the migrations list used by the runner
        if hasattr(m, "sqlite_migrations"):
            m.sqlite_migrations = [
                s.replace(bad, good) if isinstance(s, str) and bad in s else s
                for s in m.sqlite_migrations
            ]

        logger.info("dbos_sqlite_migrations_patched", sqlite_version=sqlite3.sqlite_version)
    except Exception:
        pass


# Patch BEFORE importing DBOS (which imports _migration at import time)
_patch_dbos_sqlite_migrations()

from dbos import DBOS  # noqa: E402


def init_dbos(db_path: str | Path = "exoclaw.sqlite") -> None:
    """Initialize DBOS with SQLite and launch (which auto-recovers)."""
    db_file = Path(db_path)

    # Remove stale DB with broken schema from pre-patch runs.
    if db_file.exists():
        import sqlite3

        try:
            conn = sqlite3.connect(str(db_file))
            conn.execute(
                "INSERT INTO application_versions (version_id, version_name) "
                "VALUES ('__test__', '__test__')"
            )
            conn.execute("DELETE FROM application_versions WHERE version_id='__test__'")
            conn.commit()
            conn.close()
        except Exception:
            logger.warning("dbos_removing_broken_db", db_path=str(db_file))
            db_file.unlink()

    DBOS(
        config={
            "name": "exoclaw",
            "system_database_url": f"sqlite:///{db_file}",
        }
    )
    DBOS.launch()
    logger.info("dbos_initialized", db_path=str(db_file))
