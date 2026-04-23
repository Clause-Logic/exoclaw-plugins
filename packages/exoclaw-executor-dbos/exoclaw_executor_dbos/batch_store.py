"""DBOS-backed ``BatchStore`` for ``exoclaw-subagent``.

``SubagentManager``'s default ``InMemoryBatchStore`` loses batch state on
process restart. Under a durable spawner (``DBOSSubagentSpawner``) the
subagent workflow survives restarts but any completion that lands on a
recovered process finds an empty ``_batches`` dict and silently drops.

``DBOSBatchStore`` fixes that by persisting batch state to the workspace
filesystem and wrapping every read/write in a ``@DBOS.step()``. The step
boundary means:

* Within a single workflow, replay returns the journaled snapshot and
  skips re-running the body — the publish side of the announce is
  protected from duplicate re-publishes within one workflow's recovery.
* Across workflows (e.g. two subagent children finishing near the same
  time), each child's step journaling is independent. If both workflows
  race past the threshold check, both invoke the announce callback —
  this is the at-least-once posture chosen in PR #44 for the final
  reply: prefer one duplicate message over a silent drop.

Disk layout under ``<workspace>/batches/<_safe_name(batch_id)>/``:

* ``meta.json`` — ``{batch_id, origin_channel, origin_chat_id, session_key}``.
  Written on ``register`` (first-write-wins — routing fields are stable
  within a batch by construction). ``batch_id`` is the original caller-
  supplied string so ``list_active`` can report the unsanitised value.
* ``<_safe_name(task_id)>.json`` — ``{status, label, result_path}``.
  Written on ``register`` (status ``"registered"``) and overwritten on
  ``record_completion_and_maybe_announce`` (status ``"completed"`` or
  ``"failed"``).
* ``.announced`` — sentinel file written after the announce callback
  completes. Subsequent completions that see it skip the announce.
  When the same ``batch_id`` is reused in a later run, ``register``
  deletes the whole directory first so the stale sentinel can't
  suppress the new run's announce.

Complexity: each completion reads every member file to compute totals
(O(n) per completion → O(n²) over the batch's lifetime). Fine for
openclaw's current batch sizes (≤ ~20). If batches grow, move counters
into ``meta.json`` and update them under the existing lock.

Filesystem as durable store works for single-container openclaw (the
workspace volume is persistent). The flagship Temporal port will ship
its own ``BatchStore`` against a proper DB; the generic protocol in
``exoclaw-subagent`` doesn't care which substrate either uses.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from dbos import DBOS
from exoclaw_subagent import AnnounceCallback, BatchSnapshot


def _batch_dir(workspace: Path, batch_id: str) -> Path:
    return workspace / "batches" / batch_id


def _safe_name(value: str) -> str:
    """Convert an arbitrary id to something usable as a single path
    segment. Falls back to a hex digest if the input strips empty —
    otherwise all-punctuation ids would collapse to ``""`` and all such
    batches would collide on ``<workspace>/batches/``.
    """
    cleaned = "".join(c if c.isalnum() or c in "-_" else "-" for c in value).strip("-")
    if not cleaned:
        cleaned = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return cleaned


@DBOS.step()
async def _register_step(
    workspace_str: str,
    batch_id: str,
    task_id: str,
    session_key: str | None,
    origin_channel: str,
    origin_chat_id: str,
) -> None:
    """Idempotent register. Writes meta + a per-task registered file.

    Wrapped as a DBOS step so the workspace write is journaled — replay
    within the spawning workflow returns immediately without re-touching
    the filesystem. Idempotent at the FS level too (writes overwrite).

    ``batch_id`` reuse: if a prior run under the same id is already
    announced (``.announced`` sentinel present), the whole batch dir is
    blown away before we re-register. Without this, the stale sentinel
    would suppress the next run's announce forever.
    """
    bdir = _batch_dir(Path(workspace_str), _safe_name(batch_id))
    if (bdir / ".announced").exists():
        shutil.rmtree(bdir, ignore_errors=True)
    bdir.mkdir(parents=True, exist_ok=True)

    # First-write-wins for meta — routing fields are stable within a
    # batch by construction, and rewriting per task serialises every
    # register call for no benefit.
    meta_path = bdir / "meta.json"
    if not meta_path.exists():
        meta = {
            "batch_id": batch_id,
            "origin_channel": origin_channel,
            "origin_chat_id": origin_chat_id,
            "session_key": session_key,
        }
        _atomic_write(meta_path, json.dumps(meta))

    task_path = bdir / f"{_safe_name(task_id)}.json"
    if not task_path.exists():
        _atomic_write(
            task_path,
            json.dumps({"status": "registered", "label": None, "result_path": None}),
        )


@DBOS.step()
async def _record_completion_and_decide_step(
    workspace_str: str,
    batch_id: str,
    task_id: str,
    status: str,
    label: str,
    result_path: str | None,
) -> tuple[dict, bool]:
    """Record this task's completion + snapshot the batch + decide announce.

    Returns ``(snapshot_dict, should_announce)``. Journaled — replay in
    the same workflow returns the tuple without touching disk. The
    ``announce`` callback is NOT invoked here (keeping the step pure-ish
    and serializable); the caller runs the callback after this step
    returns ``True`` and then marks the batch announced.
    """
    bdir = _batch_dir(Path(workspace_str), _safe_name(batch_id))
    if not bdir.exists():
        # No prior register — bubble up as KeyError to the manager
        # (which turns it into the ``subagent_done_orphaned`` warning).
        raise KeyError(batch_id)

    task_path = bdir / f"{_safe_name(task_id)}.json"
    _atomic_write(
        task_path,
        json.dumps({"status": status, "label": label, "result_path": result_path}),
    )

    meta_path = bdir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    members: list[dict] = []
    for f in sorted(bdir.glob("*.json")):
        if f.name == "meta.json":
            continue
        try:
            members.append(json.loads(f.read_text()))
        except Exception:
            continue

    total = len(members)
    completed = sum(1 for m in members if m.get("status") not in (None, "registered"))
    already_announced = (bdir / ".announced").exists()
    should_announce = completed >= total and not already_announced and total > 0

    snapshot = {
        "batch_id": batch_id,
        "total": total,
        "completed": completed,
        "results": [
            {
                "label": m.get("label") or "",
                "status": m.get("status") or "registered",
                "path": m.get("result_path") or "(no file)",
            }
            for m in members
            if m.get("status") not in (None, "registered")
        ],
        "origin_channel": meta.get("origin_channel", "cli"),
        "origin_chat_id": meta.get("origin_chat_id", "direct"),
        "session_key": meta.get("session_key"),
    }
    return snapshot, should_announce


@DBOS.step()
async def _mark_announced_step(workspace_str: str, batch_id: str) -> None:
    """Write the sentinel that blocks future announce attempts.

    Idempotent — ``touch`` semantics.
    """
    bdir = _batch_dir(Path(workspace_str), _safe_name(batch_id))
    sentinel = bdir / ".announced"
    sentinel.touch()


def _atomic_write(path: Path, content: str) -> None:
    """Write-then-rename so readers never see a partial file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _snapshot_from_dict(d: dict) -> BatchSnapshot:
    return BatchSnapshot(
        batch_id=d["batch_id"],
        total=d["total"],
        completed=d["completed"],
        results=list(d["results"]),
        origin_channel=d["origin_channel"],
        origin_chat_id=d["origin_chat_id"],
        session_key=d["session_key"],
    )


class DBOSBatchStore:
    """``BatchStore`` implementation whose state is durable across
    process restarts.

    Must be constructed before any ``DBOS.launch()`` call so its step
    functions are registered on the DBOS instance that will execute
    subagent workflows.
    """

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    async def register(
        self,
        batch_id: str,
        task_id: str,
        *,
        session_key: str | None,
        origin_channel: str,
        origin_chat_id: str,
    ) -> None:
        await _register_step(
            str(self._workspace),
            batch_id,
            task_id,
            session_key,
            origin_channel,
            origin_chat_id,
        )

    async def record_completion_and_maybe_announce(
        self,
        batch_id: str,
        task_id: str,
        *,
        status: str,
        label: str,
        result_path: str | None,
        announce: AnnounceCallback,
    ) -> BatchSnapshot:
        snap_dict, should_announce = await _record_completion_and_decide_step(
            str(self._workspace),
            batch_id,
            task_id,
            status,
            label,
            result_path,
        )
        snapshot = _snapshot_from_dict(snap_dict)

        if should_announce:
            # Publish FIRST, mark announced AFTER — crash between the two
            # produces a duplicate announce on replay. That matches the
            # PR #44 final-reply fix and is preferable to a silent drop.
            await announce(snapshot)
            await _mark_announced_step(str(self._workspace), batch_id)

        return snapshot

    def list_active(self) -> list[BatchSnapshot]:
        """Enumerate batches on disk that haven't been announced yet.

        Used only by ``SpawnTool``'s status action — best-effort and not
        on any hot path. Reads happen outside a DBOS workflow so they
        can't be journaled, but that's fine for human-visible status.
        """
        root = self._workspace / "batches"
        if not root.exists():
            return []
        out: list[BatchSnapshot] = []
        for bdir in sorted(root.iterdir()):
            if not bdir.is_dir() or (bdir / ".announced").exists():
                continue
            meta_path = bdir / "meta.json"
            meta: dict = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    continue
            members: list[dict] = []
            for f in sorted(bdir.glob("*.json")):
                if f.name == "meta.json":
                    continue
                try:
                    members.append(json.loads(f.read_text()))
                except Exception:
                    continue
            total = len(members)
            completed = sum(1 for m in members if m.get("status") not in (None, "registered"))
            # Prefer the original batch_id stored in meta so log/chat
            # lines stay consistent with what callers see; fall back to
            # the on-disk (sanitised) dir name for pre-upgrade batches
            # that were written before meta carried ``batch_id``.
            snapshot_batch_id = meta.get("batch_id")
            if not isinstance(snapshot_batch_id, str) or not snapshot_batch_id:
                snapshot_batch_id = bdir.name
            out.append(
                BatchSnapshot(
                    batch_id=snapshot_batch_id,
                    total=total,
                    completed=completed,
                    results=[
                        {
                            "label": m.get("label") or "",
                            "status": m.get("status") or "registered",
                            "path": m.get("result_path") or "(no file)",
                        }
                        for m in members
                        if m.get("status") not in (None, "registered")
                    ],
                    origin_channel=meta.get("origin_channel", "cli"),
                    origin_chat_id=meta.get("origin_chat_id", "direct"),
                    session_key=meta.get("session_key"),
                )
            )
        return out


__all__ = ["DBOSBatchStore"]
