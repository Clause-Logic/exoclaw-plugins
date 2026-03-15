---
name: cron
description: Schedule reminders, recurring tasks, and one-time jobs
---

# Cron Scheduling

Schedule tasks to run later or on a recurring basis. Jobs persist across restarts.

## Actions

### Add a job

One-time:
```json
{"action": "add", "message": "Check deploy status", "at": "2026-03-15T14:00:00"}
```

Recurring (cron expression):
```json
{"action": "add", "message": "Collect RSS feeds", "cron_expr": "0 * * * *", "tz": "America/New_York"}
```

Recurring (interval):
```json
{"action": "add", "message": "Health check", "every_seconds": 300}
```

### Options

- **deliver** — if true, send the job's output to the user. If false, run silently (default: true)
- **skills** — load these skills into context when the job runs: `"skills": ["rss-collector", "personal-api"]`
- **stateless** — if true, run without session history (fresh context each time)
- **tz** — IANA timezone for cron expressions (e.g. `America/New_York`, `US/Eastern`)

### List, update, remove

```json
{"action": "list"}
{"action": "update", "job_id": "abc123", "deliver": false}
{"action": "remove", "job_id": "abc123"}
```

## Cron expression format

```
┌───────────── minute (0-59)
│ ┌───────────── hour (0-23)
│ │ ┌───────────── day of month (1-31)
│ │ │ ┌───────────── month (1-12)
│ │ │ │ ┌───────────── day of week (0-7, 0=Sun, or MON-SUN)
│ │ │ │ │
* * * * *
```

Examples:
- `0 9 * * 1-5` — weekdays at 9am
- `*/30 * * * *` — every 30 minutes
- `0 7 * * *` — daily at 7am
- `0 0 1 * *` — first of every month at midnight

## Rules

- Cannot schedule new jobs from within a cron job execution
- One-time jobs (`at`) auto-delete after running
- Jobs run in the context of the channel/chat they were created from
