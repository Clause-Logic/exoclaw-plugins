---
name: workspace
description: Read, write, edit, and list files inside the agent's workspace
---

# Workspace File Tools

Four tools to manage files inside the workspace directory. All paths are
resolved relative to the workspace and may not escape it (no `..` segments).

On chip the workspace is typically a directory on the SD card mount
(e.g. `/sd/exoclaw/workspace`); on a host it's whatever directory the
firmware was launched with. Files persist across restarts.

## `read_file`

Read a file's contents. For large files use `offset` + `limit` to read a
line range instead of buffering the whole file.

```json
{"path": "notes/2026-04.md"}
{"path": "log.txt", "offset": 0, "limit": 50}
```

Returns the file content. If the file is larger than ~32 KB on chip
(~128 KB on host), the call is rejected and you must pass `offset` /
`limit`.

## `write_file`

Overwrite (or create) a file with the given content. Parent directories
are created as needed.

```json
{"path": "notes/idea.md", "content": "# Idea\n\nA new thought."}
```

## `edit_file`

Find an exact `old_text` substring and replace it with `new_text`. The
match must be unique; if `old_text` appears multiple times the call is
rejected and you should re-fetch with more context until the match is
unambiguous.

```json
{
  "path": "notes/idea.md",
  "old_text": "A new thought.",
  "new_text": "A new thought, expanded with details."
}
```

Use this for surgical edits to existing files. For wholesale rewrites
prefer `write_file`.

## `list_dir`

List a directory's contents. Files are prefixed `[f]`, directories `[d]`.

```json
{"path": "notes"}
```

## When to use

- **`read_file`** before editing — see what's there. Always read
  before `edit_file` so `old_text` matches exactly.
- **`write_file`** for fresh files or wholesale rewrites.
- **`edit_file`** for surgical changes to existing files.
- **`list_dir`** before reading — see what's in a directory you don't
  remember the layout of.
