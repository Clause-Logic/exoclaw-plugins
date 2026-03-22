"""DBOS initialization and recovery for exoclaw.

Call init_dbos() once at app startup, before any turns run.
It initializes DBOS with SQLite and recovers any incomplete
workflows from a previous run.
"""

from __future__ import annotations

from pathlib import Path

import structlog
from dbos import DBOS

logger = structlog.get_logger()


def init_dbos(db_path: str | Path = "exoclaw.sqlite") -> None:
    """Initialize DBOS and recover pending workflows.

    Args:
        db_path: Path to the SQLite database. Defaults to exoclaw.sqlite
                 in the current directory.
    """
    DBOS(
        config={
            "name": "exoclaw",
            "system_database_url": f"sqlite:///{db_path}",
        }
    )
    DBOS.launch()
    logger.info("dbos_initialized", db_path=str(db_path))

    # Recover any turns that were in progress when the process died.
    # DBOS replays completed steps and continues from the next one.
    recover = getattr(DBOS, "recover_pending_workflows", None)
    if recover is not None:
        recover()
    logger.info("dbos_recovery_complete")
