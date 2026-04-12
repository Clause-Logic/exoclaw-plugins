"""SQLite compatibility patch for DBOS.

The caller owns the DBOS lifecycle: construct ``DBOS(config=...)`` and call
``DBOS.launch()`` yourself. Importing this package applies a SQLite migration
patch needed on systems where sqlite < 3.42.0, and imports the plugin's
workflow/step modules so their decorators register with DBOS before launch.
"""

from __future__ import annotations

import sqlite3

import structlog

logger = structlog.get_logger()


def apply_sqlite_patch() -> None:
    """Fix DBOS bug: ``unixepoch('subsec')`` requires SQLite >= 3.42.0 but
    DBOS checks Python version instead.

    The migration SQL uses f-strings evaluated at import time, so we mutate
    the already-evaluated migration strings post-hoc. Safe to call whether
    or not ``dbos`` has been imported yet, as long as it runs before
    ``DBOS()`` is constructed.
    """
    if sqlite3.sqlite_version_info >= (3, 42, 0):
        return

    try:
        import dbos._migration as m

        bad = "(unixepoch('subsec') * 1000)"
        good = "(strftime('%s','now') * 1000)"

        for attr in dir(m):
            if attr.startswith("sqlite_migration_"):
                val = getattr(m, attr)
                if isinstance(val, str) and bad in val:
                    setattr(m, attr, val.replace(bad, good))

        if hasattr(m, "sqlite_migrations"):
            m.sqlite_migrations = [
                s.replace(bad, good) if isinstance(s, str) and bad in s else s
                for s in m.sqlite_migrations
            ]

        logger.info("dbos_sqlite_patched", **{"sqlite.version": sqlite3.sqlite_version})
    except Exception:
        logger.warning(
            "dbos_sqlite_patch_failed",
            **{"sqlite.version": sqlite3.sqlite_version},
            exc_info=True,
        )
