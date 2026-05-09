"""Sidecar state for ConsolidationPolicy implementations.

Each policy instance gets its own JSON sidecar file next to the session
log: ``<state_dir>/<safe_key>.consolidation.json``. The sidecar carries
the policy's view-rebuilding state — what's been summarized, the
preamble text, running token estimates — so the append-only session
log itself stays untouched by consolidation.

Migration: when ``load_state`` is called and no sidecar exists, the
loader peeks at the legacy ``<safe_key>.jsonl`` metadata header
sitting next to it. If that header carries a non-zero
``last_consolidated`` (or a ``metadata.summary``), the sidecar is
seeded from those values and persisted. After this PR's rollout,
``last_consolidated`` is never written; the sidecar is the single
source of truth.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from exoclaw._compat import Path, get_logger

from .helpers import ensure_dir, safe_filename

logger = get_logger()


_SIDECAR_VERSION = "summarizing/v1"


class ConsolidationState:
    """In-memory representation of a policy sidecar.

    Plain attribute container — no validation. Loaders and the policy
    own consistency. Serialized as JSON; format is forward-compatible
    via the ``policy`` version tag.

    Fields:
        summarized_through: absolute index in the session log up to
            which messages have been folded into ``summary``. The
            policy emits log entries from this index onwards as the
            unconsolidated tail.
        summary: rolling preamble text emitted before the tail, or
            empty if nothing has been summarized yet.
        unconsolidated_token_estimate: running estimate of tokens in
            the post-summary tail. Maintained incrementally by
            ``on_turn_complete``. Lets ``transform(budget=...)`` make
            O(1) decisions about whether further compaction is needed.
        last_updated: ISO timestamp of the last sidecar write.
    """

    def __init__(
        self,
        *,
        summarized_through: int = 0,
        summary: str = "",
        unconsolidated_token_estimate: int = 0,
        last_updated: str | None = None,
    ) -> None:
        self.summarized_through = summarized_through
        self.summary = summary
        self.unconsolidated_token_estimate = unconsolidated_token_estimate
        self.last_updated = last_updated

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": _SIDECAR_VERSION,
            "summarized_through": self.summarized_through,
            "summary": self.summary,
            "unconsolidated_token_estimate": self.unconsolidated_token_estimate,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConsolidationState:
        return cls(
            summarized_through=int(data.get("summarized_through", 0)),
            summary=str(data.get("summary", "")),
            unconsolidated_token_estimate=int(data.get("unconsolidated_token_estimate", 0)),
            last_updated=data.get("last_updated"),
        )


def sidecar_path(state_dir: Path, key: str) -> Path:
    """Compute the sidecar path for a session key. Mirrors
    ``SessionManager._get_session_path`` so sidecars sit next to
    their session JSONL when ``state_dir`` is the sessions dir."""
    safe_key = safe_filename(key.replace(":", "_"))
    return state_dir / f"{safe_key}.consolidation.json"


def _legacy_session_path(state_dir: Path, key: str) -> Path:
    """Path the JSONL session file *would* live at, if ``state_dir``
    is the sessions directory (the recommended layout). Used only by
    the migration shim in ``load_state``."""
    safe_key = safe_filename(key.replace(":", "_"))
    return state_dir / f"{safe_key}.jsonl"


def _read_legacy_boundary(jsonl_path: Path) -> tuple[int, str]:
    """Peek at a session JSONL's metadata header and return
    ``(last_consolidated, summary)``. Returns ``(0, "")`` if the
    file doesn't exist, lacks a metadata line, or fails to parse."""
    if not jsonl_path.exists():
        return 0, ""
    try:
        with open(str(jsonl_path)) as f:
            first = f.readline().strip()
        if not first or '"_type"' not in first:
            return 0, ""
        data = json.loads(first)
        if data.get("_type") != "metadata":
            return 0, ""
        last_consolidated = int(data.get("last_consolidated", 0) or 0)
        summary = str((data.get("metadata") or {}).get("summary") or "")
        return last_consolidated, summary
    except Exception:
        return 0, ""


def load_state(state_dir: Path, key: str) -> ConsolidationState:
    """Load the policy sidecar for ``key``.

    Migration: if the sidecar does not exist and a legacy session JSONL
    sits next to it with a non-zero ``last_consolidated`` (or a
    ``metadata.summary``), seed a fresh sidecar from those values and
    persist it. After this seeding, ``last_consolidated`` is never read
    again — the sidecar owns the boundary.
    """
    path = sidecar_path(state_dir, key)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ConsolidationState.from_dict(data)
        except Exception:
            logger.warning(
                "consolidation_sidecar_load_failed",
                **{"sidecar.path": str(path)},
                exc_info=True,
            )
            return ConsolidationState()

    last_consolidated, summary = _read_legacy_boundary(_legacy_session_path(state_dir, key))
    state = ConsolidationState(
        summarized_through=last_consolidated,
        summary=summary,
    )
    if last_consolidated > 0 or summary:
        try:
            save_state(state_dir, key, state)
            logger.info(
                "consolidation_sidecar_migrated",
                **{
                    "session.key": key,
                    "sidecar.summarized_through": last_consolidated,
                    "sidecar.has_summary": bool(summary),
                },
            )
        except Exception:
            logger.warning(
                "consolidation_sidecar_migrate_failed",
                **{"session.key": key},
                exc_info=True,
            )
    return state


def save_state(state_dir: Path, key: str, state: ConsolidationState) -> None:
    """Write the sidecar atomically (write-then-rename)."""
    ensure_dir(state_dir)
    state.last_updated = datetime.now().isoformat()
    path = sidecar_path(state_dir, key)
    # ``Path.with_suffix`` / ``Path.replace`` aren't on the MP
    # ``exoclaw._compat.Path`` shim (it's a Path subset, not a full
    # pathlib reimplementation). Construct the temp path via string
    # concat and rename via ``os.replace`` — both work cross-runtime.
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))


def delete_state(state_dir: Path, key: str) -> None:
    """Remove the sidecar — used when a session is cleared or deleted.
    No-op if the sidecar doesn't exist."""
    path = sidecar_path(state_dir, key)
    if path.exists():
        try:
            path.unlink()
        except Exception:
            logger.warning(
                "consolidation_sidecar_delete_failed",
                **{"session.key": key},
                exc_info=True,
            )
