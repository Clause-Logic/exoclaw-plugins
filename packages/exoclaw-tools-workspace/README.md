# exoclaw-tools-workspace

File system, shell, and web tools implementing the exoclaw `ToolBase` protocol — read/write/edit files, list directories, execute shell commands, search the web, and fetch URLs.

## Install

```
pip install exoclaw-tools-workspace
```

## Usage

```python
from pathlib import Path
from exoclaw_tools_workspace.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from exoclaw_tools_workspace.shell import ExecTool
from exoclaw_tools_workspace.web import WebSearchTool, WebFetchTool

workspace = Path("~/.nanobot/workspace").expanduser()

tools = [
    ReadFileTool(workspace=workspace),
    WriteFileTool(workspace=workspace),
    EditFileTool(workspace=workspace),
    ListDirTool(workspace=workspace),
    ExecTool(timeout=30, working_dir=str(workspace)),
    WebSearchTool(api_key="..."),   # Brave Search API key, or set BRAVE_API_KEY
    WebFetchTool(),
]
```

All tools accept an optional `allowed_dir` to restrict file operations to within a directory. `ExecTool` ships with a built-in deny-list of destructive shell patterns.
