"""DBOS initialization and recovery for exoclaw.

Call init_dbos() once at app startup to initialize DBOS with SQLite.
Call recover() after set_turn_context() to resume incomplete workflows.
"""

from __future__ import annotations

from pathlib import Path

import structlog
from dbos import DBOS

logger = structlog.get_logger()


def init_dbos(db_path: str | Path = "exoclaw.sqlite") -> None:
    """Initialize DBOS with SQLite.

    Call this once at startup, before any turns run.
    Does NOT trigger recovery — call recover() separately after
    set_turn_context() has been called.
    """
    DBOS(
        config={
            "name": "exoclaw",
            "system_database_url": f"sqlite:///{db_path}",
        }
    )
    DBOS.launch()
    logger.info("dbos_initialized", db_path=str(db_path))


def recover() -> None:
    """Recover incomplete workflows from a previous run.

    Must be called AFTER set_turn_context() so that recovered
    workflows have access to provider/conversation/tools.
    """
    recover_fn = getattr(DBOS, "recover_pending_workflows", None)
    if recover_fn is not None:
        recover_fn()
    logger.info("dbos_recovery_complete")
