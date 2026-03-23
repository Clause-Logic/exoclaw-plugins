"""DBOS initialization for exoclaw.

Call init_dbos() once at app startup, after set_turn_context().
DBOS.launch() automatically recovers any incomplete workflows.
"""

from __future__ import annotations

from pathlib import Path

import structlog
from dbos import DBOS

logger = structlog.get_logger()


def _patch_dbos_sqlite_timestamp() -> None:
    """Fix DBOS bug: unixepoch('subsec') requires SQLite >= 3.42.0 but
    DBOS checks Python version instead. Monkey-patch before migrations run.

    See: https://github.com/dbos-inc/dbos-transact-py/issues/XXX
    """
    import sqlite3

    if sqlite3.sqlite_version_info >= (3, 42, 0):
        return  # no fix needed

    try:
        import dbos._migration as m

        def _fixed() -> str:
            return "(strftime('%s','now') * 1000)"

        m.get_sqlite_timestamp_expr = _fixed
        logger.info(
            "dbos_sqlite_timestamp_patched",
            sqlite_version=sqlite3.sqlite_version,
        )
    except Exception:
        pass  # if DBOS internals change, don't crash


def init_dbos(db_path: str | Path = "exoclaw.sqlite") -> None:
    """Initialize DBOS with SQLite and launch (which auto-recovers).

    Always starts with a fresh database. Recovery across deploys is
    not useful since the application code (and thus workflow definitions)
    changes with each deploy. Within a single run, DBOS journals steps
    and can recover from mid-turn crashes.
    """
    _patch_dbos_sqlite_timestamp()

    db_file = Path(db_path)

    # Remove stale DB with broken schema from pre-patch runs.
    # Once the patch is applied and a clean DB is created, this
    # block becomes a no-op (the schema will be correct).
    if db_file.exists():
        import sqlite3

        try:
            conn = sqlite3.connect(str(db_file))
            # Test if the DEFAULT works — if not, the schema is broken
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
