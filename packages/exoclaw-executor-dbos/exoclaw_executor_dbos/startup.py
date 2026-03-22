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
    db_url = f"sqlite:///{db_path}"
    try:
        DBOS(
            config={
                "name": "exoclaw",
                "system_database_url": db_url,
            }
        )
        DBOS.launch()
        logger.info("dbos_initialized", db_path=str(db_path))
    except Exception:
        # If the DB is corrupt from a previous failed init, delete and retry
        db_file = Path(db_path)
        if db_file.exists():
            logger.warning("dbos_init_failed_retrying", db_path=str(db_path))
            db_file.unlink()
            DBOS(
                config={
                    "name": "exoclaw",
                    "system_database_url": db_url,
                }
            )
            DBOS.launch()
            logger.info("dbos_initialized_after_reset", db_path=str(db_path))
        else:
            raise
