# exoclaw-github

GitHub Actions channel for [exoclaw](https://github.com/stephensolka/exoclaw).

Runs the exoclaw agent stack inside a GitHub Actions workflow, using issues, PR comments, and `workflow_dispatch` as the inbound channel and GitHub comments as the outbound channel. Session history is persisted to a dedicated `bot-state` branch.

## Usage

```python
from exoclaw_github import GitHubChannel

channel = GitHubChannel(
    token="ghp_...",       # or set GITHUB_TOKEN env var
    trigger="@exoclaw",    # only respond when this appears in comments (None = all)
    respond_to_issues_opened=True,
    respond_to_prs_opened=False,
)
```

## Supported events

| Event | Default behaviour |
|---|---|
| `issues` (opened) | Always respond |
| `issue_comment` (created) | Respond if trigger word present |
| `pull_request` (opened) | Off by default |
| `workflow_dispatch` | Always respond |

## Session state

Sessions are keyed as `github:issue:{number}` or `github:pr:{number}`. When used with `exoclaw-conversation`, history is stored in `~/.nanobot/workspace/sessions/`. Check out the `bot-state` branch there before running and commit it back afterwards to persist state across workflow runs.
