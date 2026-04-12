"""exoclaw-executor-dbos — durable agent-turn execution via DBOS.

The caller owns the DBOS lifecycle. Typical wiring:

    import exoclaw_executor_dbos  # registers workflows/steps, patches sqlite
    from dbos import DBOS

    DBOS(config={"name": "myapp", "system_database_url": "sqlite:///app.sqlite"})
    # ...define your own @DBOS.workflow / @DBOS.scheduled here...
    DBOS.launch()
"""

from .startup import apply_sqlite_patch

apply_sqlite_patch()

from .executor import DBOSExecutor  # noqa: E402
from .turn import run_durable_turn, set_loop_context  # noqa: E402

__all__ = ["DBOSExecutor", "apply_sqlite_patch", "run_durable_turn", "set_loop_context"]
