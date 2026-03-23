"""DBOS initialization for exoclaw.

Call init_dbos() once at app startup, after set_turn_context().
DBOS.launch() automatically recovers any incomplete workflows.
"""

from __future__ import annotations

from pathlib import Path

import structlog
from dbos import DBOS

logger = structlog.get_logger()


def init_dbos(db_path: str | Path = "exoclaw.sqlite") -> None:
    """Initialize DBOS with SQLite and launch (which auto-recovers).

    Args:
        db_path: Path to the SQLite database. Defaults to exoclaw.sqlite
                 in the current directory.
    """
    db_file = Path(db_path)
    db_url = f"sqlite:///{db_file}"

    # Validate the existing DB is usable. If it's corrupt or from an
    # incompatible DBOS version, remove it so init starts clean.
    # This is safe because DBOS recovery only matters within the same
    # application version — a new deploy invalidates old workflows anyway.
    if db_file.exists():
        try:
            import sqlite3

            conn = sqlite3.connect(str(db_file))
            conn.execute("SELECT 1 FROM application_versions LIMIT 1")
            conn.close()
        except Exception:
            logger.warning("dbos_stale_db_removed", db_path=str(db_file))
            db_file.unlink()

    DBOS(
        config={
            "name": "exoclaw",
            "system_database_url": db_url,
        }
    )
    DBOS.launch()
    logger.info("dbos_initialized", db_path=str(db_file))
