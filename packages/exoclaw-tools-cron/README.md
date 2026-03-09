# exoclaw-tools-cron

Cron scheduling tool and service for exoclaw — lets the agent schedule one-time, recurring, and cron-expression-based tasks that persist across restarts.

## Install

```
pip install exoclaw-tools-cron
```

## Usage

```python
from pathlib import Path
from exoclaw_tools_cron.service import CronService
from exoclaw_tools_cron.tool import CronTool

cron_service = CronService(
    store_path=Path("~/.nanobot/workspace/cron.json").expanduser(),
    on_job=my_job_handler,   # async callable that receives a CronJob
)
await cron_service.start()

cron_tool = CronTool(cron_service=cron_service)
# Register cron_tool with the agent's tool registry
```

The agent uses `CronTool` to add/list/remove/update jobs. `CronService` persists jobs to disk in JSON and reloads automatically if the file is modified externally. Supports `every_seconds`, `cron_expr` (with IANA timezone), and `at` (ISO datetime) schedules.
