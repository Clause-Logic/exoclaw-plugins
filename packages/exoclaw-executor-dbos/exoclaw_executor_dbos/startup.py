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

    Always starts with a fresh database. Recovery across deploys is
    not useful since the application code (and thus workflow definitions)
    changes with each deploy. Within a single run, DBOS journals steps
    and can recover from mid-turn crashes.

    Args:
        db_path: Path to the SQLite database. Defaults to exoclaw.sqlite
                 in the current directory.
    """
    db_file = Path(db_path)

    # Always start fresh — avoids schema compat issues between DBOS versions
    # and stale workflow state from previous deploys.
    if db_file.exists():
        db_file.unlink()

    DBOS(
        config={
            "name": "exoclaw",
            "system_database_url": f"sqlite:///{db_file}",
        }
    )
    DBOS.launch()
    logger.info("dbos_initialized", db_path=str(db_file))
