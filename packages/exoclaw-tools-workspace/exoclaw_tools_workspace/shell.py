"""Shell execution tool."""

import asyncio
import codecs
import os
import re
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from exoclaw.agent.tools.protocol import ToolBase
from exoclaw.utils import create_isolated_task


class ExecTool(ToolBase):
    """Tool to execute shell commands."""

    def __init__(
        self,
        timeout: int = 10,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
        path_append: str = "",
    ):
        self.timeout = timeout
        self.working_dir = working_dir
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",  # rm -r, rm -rf, rm -fr
            r"\bdel\s+/[fq]\b",  # del /f, del /q
            r"\brmdir\s+/s\b",  # rmdir /s
            r"(?:^|[;&|]\s*)format\b",  # format (as standalone command only)
            r"\b(mkfs|diskpart)\b",  # disk operations
            r"\bdd\s+if=",  # dd
            r">\s*/dev/sd",  # write to disk
            r"\b(shutdown|reboot|poweroff)\b",  # system power
            r":\(\)\s*\{.*\};\s*:",  # fork bomb
        ]
        self.allow_patterns = allow_patterns or []
        self.restrict_to_workspace = restrict_to_workspace
        self.path_append = path_append

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Execute a shell command and return its output. Use with caution."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"},
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command",
                },
            },
            "required": ["command"],
        }

    async def execute(self, command: str, working_dir: str | None = None, **kwargs: Any) -> str:
        cwd = working_dir or self.working_dir or os.getcwd()
        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return guard_error

        env = os.environ.copy()
        if self.path_append:
            env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
            except asyncio.TimeoutError:
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                return f"Error: Command timed out after {self.timeout} seconds"

            output_parts = []

            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))

            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")

            if process.returncode != 0:
                output_parts.append(f"\nExit code: {process.returncode}")

            result = "\n".join(output_parts) if output_parts else "(no output)"

            max_len = 10000
            if len(result) > max_len:
                result = result[:max_len] + f"\n... (truncated, {len(result) - max_len} more chars)"

            return result

        except Exception as e:
            return f"Error executing command: {str(e)}"

    async def execute_streaming(
        self, command: str, working_dir: str | None = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        """Step D opt-in: stream subprocess output as it arrives,
        rather than holding the full ``stdout`` / ``stderr`` in memory
        until the process exits.

        Same safety guard, working-dir, env, and timeout semantics as
        :meth:`execute` — but truncation is dropped (the whole point
        of streaming is to support multi-MB output) and output is
        yielded in bounded byte-chunks as the subprocess produces it.
        The executor drains the iterator into a per-turn scratch file,
        so transient Python heap pressure is bounded by one read
        chunk at a time, not by the full output size.

        Timeout enforcement is deadline-based: the total wall time
        from process start to finish is capped at ``self.timeout``,
        same as the inline path. Per-read waits use the remaining
        deadline so a hung-with-no-output process trips the timeout
        reliably (the inline path's ``process.communicate()`` would
        have done the same).

        ``stdout`` and ``stderr`` are drained **concurrently** — each
        pipe gets its own consumer task pushing decoded chunks into a
        shared queue. Sequential drain (``stdout`` first, then
        ``stderr``) deadlocks if the child writes enough to ``stderr``
        to fill the OS pipe buffer (~64 KB on Linux) while ``stdout``
        is still being drained: the child blocks on the ``stderr``
        write, ``stdout`` never reaches EOF, and the iterator hangs
        until the deadline.
        """
        cwd = working_dir or self.working_dir or os.getcwd()
        guard_error = self._guard_command(command, cwd)
        if guard_error:
            yield guard_error
            return

        env = os.environ.copy()
        if self.path_append:
            env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
        except Exception as e:
            yield f"Error executing command: {e}"
            return

        deadline = time.monotonic() + self.timeout
        timed_out = False

        # Tagged-chunk queue: ``(origin, chunk)`` where ``chunk=None``
        # is the EOF sentinel for that origin. Bounded queue size
        # provides natural backpressure — if the consumer (the agent
        # loop / scratch-file write) is slow, the producer task
        # awaits ``put`` and the OS pipe buffer absorbs the rest.
        queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue(maxsize=64)

        async def _consume(reader: asyncio.StreamReader | None, origin: str) -> None:
            """Read fixed byte chunks from a pipe and push decoded
            text into the shared queue.

            **Fixed byte chunks** (not lines) — a single 2 MB line of
            unformatted JSON or minified HTML would otherwise force
            a 2 MB allocation in ``readline``, defeating streaming.
            Each chunk feeds an incremental UTF-8 decoder so partial
            codepoints at chunk boundaries buffer until the next
            chunk completes them. ``errors='replace'`` swaps truly
            invalid sequences for U+FFFD rather than crashing.
            """
            try:
                if reader is None:
                    return
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return
                    try:
                        chunk = await asyncio.wait_for(reader.read(8192), timeout=remaining)
                    except asyncio.TimeoutError:
                        return
                    if not chunk:
                        tail = decoder.decode(b"", final=True)
                        if tail:
                            await queue.put((origin, tail))
                        return
                    text = decoder.decode(chunk)
                    if text:
                        await queue.put((origin, text))
            finally:
                # Always signal EOF for this origin so the consumer
                # knows when both pipes are done.
                await queue.put((origin, None))

        stdout_task = create_isolated_task(_consume(process.stdout, "stdout"))
        stderr_task = create_isolated_task(_consume(process.stderr, "stderr"))

        done = {"stdout": False, "stderr": False}
        stderr_prefix_emitted = False

        try:
            while not (done["stdout"] and done["stderr"]):
                origin, chunk = await queue.get()
                if chunk is None:
                    done[origin] = True
                    continue
                if origin == "stderr" and not stderr_prefix_emitted:
                    yield "STDERR:\n" + chunk
                    stderr_prefix_emitted = True
                else:
                    yield chunk

            timed_out = time.monotonic() >= deadline and process.returncode is None
            if not timed_out:
                try:
                    rc = await asyncio.wait_for(
                        process.wait(), timeout=max(0.1, deadline - time.monotonic())
                    )
                except asyncio.TimeoutError:
                    timed_out = True
                else:
                    if rc != 0:
                        yield f"\nExit code: {rc}"
        finally:
            if process.returncode is None:
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
            # Both consumer tasks should already be done (their
            # readers see EOF when the process exits / is killed),
            # but await them so there are no orphan tasks. Cancel
            # first if for some reason they're still running, e.g.
            # the deadline tripped mid-read with an unflushed pipe.
            for task in (stdout_task, stderr_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

        if timed_out:
            yield f"\nError: Command timed out after {self.timeout} seconds"

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """Best-effort safety guard for potentially destructive commands."""
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Error: Command blocked by safety guard (not in allowlist)"

        if self.restrict_to_workspace:
            if "..\\" in cmd or "../" in cmd:
                return "Error: Command blocked by safety guard (path traversal detected)"

            cwd_path = Path(cwd).resolve()

            for raw in self._extract_absolute_paths(cmd):
                try:
                    p = Path(raw.strip()).resolve()
                except Exception:
                    continue
                if p.is_absolute() and cwd_path not in p.parents and p != cwd_path:
                    return "Error: Command blocked by safety guard (path outside working dir)"

        return None

    @staticmethod
    def _extract_absolute_paths(command: str) -> list[str]:
        win_paths = re.findall(r"[A-Za-z]:\\[^\s\"'|><;]+", command)
        posix_paths = re.findall(r"(?:^|[\s|>])(/[^\s\"'>]+)", command)
        return win_paths + posix_paths
