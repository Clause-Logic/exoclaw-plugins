"""File, shell, and web tools for exoclaw.

The ``filesystem`` module is cross-runtime — same 4-tool surface
(``read_file``, ``write_file``, ``edit_file``, ``list_dir``) on both
CPython and MicroPython, with chip-friendly defaults (32 KB cap,
no ``difflib`` dependency, ``_compat.Path`` instead of ``pathlib``).

The ``shell`` and ``web`` modules are CPython-only:

- ``shell.ExecTool`` needs ``subprocess`` (no shell on a chip).
- ``web.WebSearchTool`` / ``WebFetchTool`` need ``httpx`` + ``urllib``
  + ``structlog`` and re-rendering with ``html`` — out of scope for
  the chip path right now (firmware uses ``exoclaw.http`` directly
  for the LLM call).

They aren't imported at package-import time so ``import
exoclaw_tools_workspace`` works on the chip without dragging the
host-only deps in. Callers that need them on CPython import from
the submodule directly:

    from exoclaw_tools_workspace.shell import ExecTool
    from exoclaw_tools_workspace.web import WebFetchTool, WebSearchTool

The cross-runtime filesystem tools are re-exported here.
"""

from exoclaw_tools_workspace.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)

__all__ = [
    "EditFileTool",
    "ListDirTool",
    "ReadFileTool",
    "WriteFileTool",
]
