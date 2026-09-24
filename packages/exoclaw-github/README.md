# exoclaw-github

GitHub Actions channel for [exoclaw](https://github.com/stephensolka/exoclaw).

Runs the exoclaw agent stack inside a GitHub Actions workflow, using issues, PR comments, and `workflow_dispatch` as the inbound channel and GitHub comments as the outbound channel. Session history is persisted to a dedicated `bot-state` branch by the supplied workflow template.

## Usage

```python
from exoclaw_github import GitHubChannel

channel = GitHubChannel(
    token="...",           # or set GITHUB_TOKEN env var
    trigger="@exoclaw",    # only respond when this appears in comments (None = all)
    respond_to_issues_opened=True,
    respond_to_prs_opened=False,
)
```

## Configuring an agent's surface

The following settings are optional. Unset settings keep the existing event,
skill, and tool behavior. Comma-separated allowlists are exact matches; an
explicitly empty list denies everything in that category.

| Environment variable | Effect |
|---|---|
| `EXOCLAW_ALLOWED_EVENTS` | GitHub event names accepted by the channel, such as `issues` or `issue_comment`. |
| `EXOCLAW_ISSUE_LABEL` | Exact label required for an `issues.opened` event. |
| `EXOCLAW_ISSUE_AUTHOR_ASSOCIATIONS` | Accepted `author_association` values for an `issues.opened` event. |
| `EXOCLAW_SKILLS_DIR` | Deployment skill directory; relative paths resolve inside the checked-out repository. |
| `EXOCLAW_ALLOWED_SKILLS` | Names of skills visible to the agent. Missing or unavailable names fail startup. When `EXOCLAW_SKILLS_DIR` is set, each name must resolve there. |
| `EXOCLAW_ALLOWED_TOOLS` | Names of tools registered with the agent. Unknown names fail startup. |

The same settings can be passed to `create()` as `allowed_events`,
`issue_label`, `issue_author_associations`, `skills_dir`, `allowed_skills`,
and `allowed_tools`. Event and issue filters can also be passed directly to
`GitHubChannel`. Available tool names are `read_file`, `write_file`,
`edit_file`, `list_dir`, `exec`, `github_review`, `github_label`,
`github_pr_diff`, `github_issue`, `github_reaction`, `github_file`,
`github_checks`, `github_search`, and `load_skill` when skill configuration is
provided.

When `EXOCLAW_ALLOWED_SKILLS` is nonempty, include `load_skill` in
`EXOCLAW_ALLOWED_TOOLS`. A same-named skill in agent state cannot silently
shadow a skill selected from `EXOCLAW_SKILLS_DIR`; startup fails instead.

The channel uses `GITHUB_TOKEN` to post its response. Excluding `exec` and
GitHub API tools from `EXOCLAW_ALLOWED_TOOLS` keeps those capabilities out of
the agent's tool registry while preserving the channel response.

## Supported events

| Event | Default behaviour |
|---|---|
| `issues` (opened) | Always respond |
| `issue_comment` (created) | Respond if trigger word present |
| `pull_request` (opened) | Off by default |
| `workflow_dispatch` | Always respond |

## Session state

Sessions are keyed as `github:issue:{number}` or `github:pr:{number}`. When used with `exoclaw-conversation`, history is stored in `~/.nanobot/workspace/sessions/`. Check out the `bot-state` branch there before running and commit it back afterwards to persist state across workflow runs.
