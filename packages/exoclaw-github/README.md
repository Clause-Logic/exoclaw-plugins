# exoclaw-github

GitHub Actions channel for [exoclaw](https://github.com/stephensolka/exoclaw).

Runs the exoclaw agent stack inside a GitHub Actions workflow, using issues, PR comments, and `workflow_dispatch` as the inbound channel and GitHub comments as the outbound channel.

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

## Model-issue intake

The ordinary GitHub channel behavior stays unchanged unless the model-issue
gate is enabled. Set `EXOCLAW_MODEL_ISSUE_GATE=true` (or pass
`model_issue_gate=True` to `create()` / `GitHubChannel`) to accept only an
`issues.opened` event whose issue has the exact `model` label and whose
`author_association` is one of `CONTRIBUTOR`, `MEMBER`, `OWNER`, or
`COLLABORATOR`. Comments, pull requests, workflow dispatches, other labels,
and other associations do not start a turn in that profile.

The gate can be configured with `EXOCLAW_MODEL_ISSUE_LABEL` and
`EXOCLAW_MODEL_ISSUE_AUTHOR_ASSOCIATIONS` (a comma-separated list), or the
corresponding constructor arguments. For a gated turn, the agent registry has
only read, write, edit, and directory file tools. It has no shell tool and no
GitHub API tools; the channel alone posts the final response to the triggering
issue.

`create()` discovers checked-out repository skills at
`<repo>/.agents/skills`. Gated model issues expose the reviewed
`decisionbench-add-model` skill, which is loaded on every turn when its
frontmatter has `metadata: {"exoclaw": {"always": true}}`.

## Supported events

| Event | Default behaviour |
|---|---|
| `issues` (opened) | Always respond |
| `issue_comment` (created) | Respond if trigger word present |
| `pull_request` (opened) | Off by default |
| `workflow_dispatch` | Always respond |

When the model-issue gate is enabled, only the accepted `issues` event above
is eligible.

## Session state

Sessions are keyed as `github:issue:{number}` or `github:pr:{number}`. When
used with `exoclaw-conversation`, history is stored in
`~/.nanobot/workspace/sessions/`. A model-issue workflow can leave that state
ephemeral and keep repository contents read-only until its separate PR step.
