"""Skills loader for agent capabilities."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from exoclaw._compat import IS_MICROPYTHON, Path, is_executable, which

# Default builtin skills directory — none bundled in this package
BUILTIN_SKILLS_DIR: Path | None = None

# Standard tool definition for load_skill — consumers can include this in their
# tool list so the agent can dynamically activate skills listed in <skills>.
LOAD_SKILL_TOOL_DEF: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "load_skill",
        "description": (
            "Load a skill by name. Returns the skill instructions and "
            "activates its tools for use in this conversation. Call this "
            "when a skill from <skills> is relevant to the user's request."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill name from the <skills> summary.",
                }
            },
            "required": ["name"],
        },
    },
}


if not IS_MICROPYTHON:  # pragma: no cover (micropython)
    from dataclasses import dataclass, field

    @dataclass
    class LoadSkillResult:
        """Result of activating a skill via :meth:`SkillsLoader.activate_skill`."""

        content: str
        tool_names: list[str] = field(default_factory=list)

    @dataclass
    class AgentHook:
        """An agent hook defined by a markdown file in a skill's hooks directory.

        Agent hooks are ``.md`` files at ``skills/{name}/hooks/exoclaw/{hook_name}.md``.
        The markdown content is the prompt for a fire-and-forget agent turn that
        reacts to the lifecycle event.  Frontmatter controls which tools/skills
        the hook turn has access to.

        Attributes:
            skill_name: The skill that owns this hook.
            hook_name: The lifecycle event (e.g. ``agent_end``).
            prompt: The markdown body (frontmatter stripped) — used as the agent prompt.
            tools: Tool names from frontmatter ``tools:`` field, or empty to inherit.
            skills: Skill names from frontmatter ``skills:`` field, or empty to inherit.
        """

        skill_name: str
        hook_name: str
        prompt: str
        tools: list[str] = field(default_factory=list)
        skills: list[str] = field(default_factory=list)

else:  # pragma: no cover (cpython)

    class LoadSkillResult:
        """MicroPython fallback — plain class with hand-written
        ``__init__``. Same shape as the CPython ``@dataclass`` branch
        above."""

        def __init__(
            self,
            content: str,
            tool_names: list[str] | None = None,
        ) -> None:
            self.content = content
            self.tool_names = tool_names if tool_names is not None else []

    class AgentHook:
        """MicroPython fallback — plain class with hand-written
        ``__init__``. Same shape as the CPython ``@dataclass`` branch
        above."""

        def __init__(
            self,
            skill_name: str,
            hook_name: str,
            prompt: str,
            tools: list[str] | None = None,
            skills: list[str] | None = None,
        ) -> None:
            self.skill_name = skill_name
            self.hook_name = hook_name
            self.prompt = prompt
            self.tools = tools if tools is not None else []
            self.skills = skills if skills is not None else []


class SkillsLoader:
    """
    Loader for agent skills.

    Skills are markdown files (SKILL.md) that teach the agent how to use
    specific tools or perform certain tasks.

    Package skills can provide:
        - {name, content}: SKILL.md content only
        - {name, content, path}: full skill directory with hooks
    """

    def __init__(
        self,
        workspace: Path,
        builtin_skills_dir: Path | None = None,
        skill_packages: list[str] | None = None,
    ):
        self.workspace = workspace
        self.workspace_skills = workspace / "skills"
        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR
        self._package_skills: dict[str, str] = {}
        self._package_skill_dirs: dict[str, Path] = {}
        if skill_packages:
            self._load_package_skills(skill_packages)

    def _load_package_skills(self, packages: list[str]) -> None:
        """Load skills from installed Python packages via entry points.

        Entry points return a dict with:
            name: skill name
            content: SKILL.md content
            path (optional): Path to skill directory containing hooks/, SKILL.md, etc.
        """
        import importlib.metadata

        # Build a mapping from normalized package name to its entry points
        pkg_ep_map: dict[str, list[importlib.metadata.EntryPoint]] = {}
        for dist in importlib.metadata.distributions():
            dist_name = dist.metadata["Name"]
            if dist_name is None:
                continue
            normalized = dist_name.replace("-", "_").lower()
            eps = dist.entry_points
            for ep in eps:
                if ep.group == "exoclaw.skills":
                    pkg_ep_map.setdefault(normalized, []).append(ep)

        for pkg in packages:
            normalized_pkg = pkg.replace("-", "_").lower()
            for ep in pkg_ep_map.get(normalized_pkg, []):
                try:
                    loader_fn = ep.load()
                    result = loader_fn()
                    if isinstance(result, dict) and "name" in result and "content" in result:
                        self._package_skills[result["name"]] = result["content"]
                        if "path" in result:
                            self._package_skill_dirs[result["name"]] = Path(result["path"])
                except Exception:
                    pass

    @property
    def _all_skill_dirs(self) -> list[tuple[Path, str]]:
        """Return all skill directories as (dir, source) pairs for hook scanning."""
        dirs: list[tuple[Path, str]] = []
        # Workspace (highest priority)
        if self.workspace_skills.exists():
            for d in self.workspace_skills.iterdir():
                if d.is_dir():
                    dirs.append((d, "workspace"))
        # Builtin
        if self.builtin_skills and self.builtin_skills.exists():
            for d in self.builtin_skills.iterdir():
                if d.is_dir():
                    dirs.append((d, "builtin"))
        # Package
        for name, path in self._package_skill_dirs.items():
            if path.is_dir():
                dirs.append((path, "package"))
        return dirs

    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        """
        List all available skills.

        Args:
            filter_unavailable: If True, filter out skills with unmet requirements.

        Returns:
            List of skill info dicts with 'name', 'path', 'source'.
        """
        skills = []

        # Workspace skills (highest priority)
        if self.workspace_skills.exists():
            for skill_dir in self.workspace_skills.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists():
                        skills.append(
                            {"name": skill_dir.name, "path": str(skill_file), "source": "workspace"}
                        )

        # Built-in skills
        if self.builtin_skills and self.builtin_skills.exists():
            for skill_dir in self.builtin_skills.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists() and not any(s["name"] == skill_dir.name for s in skills):
                        skills.append(
                            {"name": skill_dir.name, "path": str(skill_file), "source": "builtin"}
                        )

        # Package skills (lowest priority — workspace and builtin override)
        for name in self._package_skills:
            if not any(s["name"] == name for s in skills):
                path = (
                    str(self._package_skill_dirs[name] / "SKILL.md")
                    if name in self._package_skill_dirs
                    else f"package:{name}"
                )
                skills.append({"name": name, "path": path, "source": "package"})

        # Filter by requirements
        if filter_unavailable:
            return [s for s in skills if self._check_requirements(self._get_skill_meta(s["name"]))]
        return skills

    def load_skill(self, name: str) -> str | None:
        """
        Load a skill by name.

        Args:
            name: Skill name (directory name).

        Returns:
            Skill content or None if not found.
        """
        # Check workspace first
        workspace_skill = self.workspace_skills / name / "SKILL.md"
        if workspace_skill.exists():
            return workspace_skill.read_text(encoding="utf-8")

        # Check built-in
        if self.builtin_skills:
            builtin_skill = self.builtin_skills / name / "SKILL.md"
            if builtin_skill.exists():
                return builtin_skill.read_text(encoding="utf-8")

        # Check package skills
        if name in self._package_skills:
            return self._package_skills[name]

        return None

    def activate_skill(self, name: str) -> LoadSkillResult:
        """Load a skill's content and resolve its declared tool names.

        This is the handler behind the ``load_skill`` tool.  Consumers should
        call this when the LLM invokes ``load_skill`` and then merge the
        returned ``tool_names`` into the active optional tools set so subsequent
        LLM calls include them.

        Returns:
            A :class:`LoadSkillResult` with the skill content (or an error
            message) and the list of tool names declared by the skill.
        """
        content = self.load_skill(name)
        if content is None:
            return LoadSkillResult(content=f"Skill '{name}' not found.")
        tool_names = list(self.get_tools_for_skills([name]))
        return LoadSkillResult(content=content, tool_names=tool_names)

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """
        Load specific skills for inclusion in agent context.

        Args:
            skill_names: List of skill names to load.

        Returns:
            Formatted skills content.
        """
        parts = []
        for name in skill_names:
            content = self.load_skill(name)
            if content:
                content = self._strip_frontmatter(content)
                parts.append(f"### Skill: {name}\n\n{content}")

        return "\n\n---\n\n".join(parts) if parts else ""

    def build_skills_summary(self, only: set[str] | None = None) -> str:
        """
        Build a summary of skills (name, description, path, availability).

        Args:
            only: If provided, restrict the summary to skills whose name is in this set.

        Returns:
            XML-formatted skills summary.
        """
        all_skills = self.list_skills(filter_unavailable=False)
        if only is not None:
            all_skills = [s for s in all_skills if s["name"] in only]
        if not all_skills:
            return ""

        def escape_xml(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines = ["<skills>"]
        for s in all_skills:
            name = escape_xml(s["name"])
            path = s["path"]
            desc = escape_xml(self._get_skill_description(s["name"]))
            skill_meta = self._get_skill_meta(s["name"])
            available = self._check_requirements(skill_meta)

            lines.append(f'  <skill available="{str(available).lower()}">')
            lines.append(f"    <name>{name}</name>")
            lines.append(f"    <description>{desc}</description>")
            lines.append(f"    <location>{path}</location>")

            if not available:
                missing = self._get_missing_requirements(skill_meta)
                if missing:
                    lines.append(f"    <requires>{escape_xml(missing)}</requires>")

            lines.append("  </skill>")
        lines.append("</skills>")

        return "\n".join(lines)

    def _get_missing_requirements(self, skill_meta: dict[str, Any]) -> str:
        """Get a description of missing requirements."""
        missing = []
        requires = skill_meta.get("requires", {})
        for b in requires.get("bins", []):
            if not which(b):
                missing.append(f"CLI: {b}")
        for env in requires.get("env", []):
            if not os.environ.get(env):
                missing.append(f"ENV: {env}")
        return ", ".join(missing)

    def _get_skill_description(self, name: str) -> str:
        """Get the description of a skill from its frontmatter."""
        meta = self.get_skill_metadata(name)
        if meta and meta.get("description"):
            return str(meta["description"])
        return name

    def _strip_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from markdown content."""
        if content.startswith("---"):
            match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
            if match:
                return content[match.end() :].strip()
        return content

    def _parse_exoclaw_metadata(self, raw: str) -> dict[str, Any]:
        """Parse skill metadata JSON from frontmatter (supports exoclaw, nanobot, openclaw keys)."""
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                result = data.get("exoclaw", data.get("nanobot", data.get("openclaw", {})))
                return dict(result) if isinstance(result, dict) else {}
            return {}
        except (ValueError, json.JSONDecodeError, TypeError):
            # MP's ``json.loads`` raises plain ``ValueError``;
            # CPython's ``JSONDecodeError`` is a ``ValueError``
            # subclass. Catch the union for cross-runtime safety.
            return {}

    def _check_requirements(self, skill_meta: dict[str, Any]) -> bool:
        """Check if skill requirements are met (bins, env vars)."""
        requires = skill_meta.get("requires", {})
        for b in requires.get("bins", []):
            if not which(b):
                return False
        for env in requires.get("env", []):
            if not os.environ.get(env):
                return False
        return True

    def _get_skill_meta(self, name: str) -> dict[str, Any]:
        """Get exoclaw metadata for a skill (cached in frontmatter)."""
        meta = self.get_skill_metadata(name) or {}
        return self._parse_exoclaw_metadata(meta.get("metadata", ""))

    def get_bootstrap_injections(self) -> list[str]:
        """Return content from hooks/exoclaw/bootstrap.md for all installed skills."""
        results = []
        seen: set[str] = set()
        for skill_dir, _source in self._all_skill_dirs:
            if skill_dir.name in seen:
                continue
            seen.add(skill_dir.name)
            for hook_path in (
                skill_dir / "hooks" / "exoclaw" / "bootstrap.md",
                skill_dir / "hooks" / "nanobot" / "bootstrap.md",
            ):
                if hook_path.exists():
                    content = hook_path.read_text(encoding="utf-8").strip()
                    if content:
                        results.append(content)
                    break
        return results

    def get_skill_hook_scripts(self, hook_name: str) -> list[Path]:
        """Return paths to executable hook scripts named hook_name across all installed skills."""
        results = []
        seen: set[str] = set()
        for skill_dir, _source in sorted(self._all_skill_dirs, key=lambda x: x[0].name):
            if skill_dir.name in seen:
                continue
            seen.add(skill_dir.name)
            for hook_file in (
                skill_dir / "hooks" / "exoclaw" / hook_name,
                skill_dir / "hooks" / "nanobot" / hook_name,
            ):
                if hook_file.exists() and is_executable(hook_file):
                    results.append(hook_file)
                    break
        return results

    def get_agent_hooks(self, hook_name: str) -> list[AgentHook]:
        """Return agent hooks for a lifecycle event across all installed skills.

        Agent hooks are ``.md`` files at ``hooks/exoclaw/{hook_name}.md`` inside
        a skill directory.  Each file becomes a fire-and-forget agent turn when
        the event fires.  Frontmatter ``tools:`` and ``skills:`` control what
        the hook turn has access to (empty = inherit from parent).

        Args:
            hook_name: Lifecycle event name (e.g. ``agent_end``).

        Returns:
            List of :class:`AgentHook` instances, one per skill that defines
            the hook.  Order matches skill priority (workspace > builtin > package).
        """
        results: list[AgentHook] = []
        seen: set[str] = set()
        for skill_dir, _source in self._all_skill_dirs:
            if skill_dir.name in seen:
                continue
            seen.add(skill_dir.name)
            for hook_path in (
                skill_dir / "hooks" / "exoclaw" / f"{hook_name}.md",
                skill_dir / "hooks" / "nanobot" / f"{hook_name}.md",
            ):
                if hook_path.exists():
                    raw = hook_path.read_text(encoding="utf-8").strip()
                    if not raw:
                        break
                    # Parse frontmatter
                    tools: list[str] = []
                    skills: list[str] = []
                    prompt = raw
                    if raw.startswith("---"):
                        match = re.match(r"^---\n(.*?)\n---\n?", raw, re.DOTALL)
                        if match:
                            prompt = raw[match.end() :].strip()
                            for line in match.group(1).splitlines():
                                if ":" in line:
                                    key, value = line.split(":", 1)
                                    key = key.strip()
                                    if key == "tools":
                                        tools = [t.strip() for t in value.split(",") if t.strip()]
                                    elif key == "skills":
                                        skills = [s.strip() for s in value.split(",") if s.strip()]
                    results.append(
                        AgentHook(
                            skill_name=skill_dir.name,
                            hook_name=hook_name,
                            prompt=prompt,
                            tools=tools,
                            skills=skills,
                        )
                    )
                    break
        return results

    def get_tools_for_skills(self, skill_names: list[str]) -> set[str]:
        """Return the union of tool names declared by the given skills.

        Skills list required optional tools in their frontmatter:

            tools: mcp_sentry_get_issues, mcp_sentry_resolve_issue

        The value is a comma-separated list of tool names.  Skills that
        don't declare a ``tools`` key contribute nothing.
        """
        result: set[str] = set()
        for name in skill_names:
            meta = self.get_skill_metadata(name) or {}
            raw = meta.get("tools", "")
            if raw:
                for tool_name in raw.split(","):
                    stripped = tool_name.strip()
                    if stripped:
                        result.add(stripped)
        return result

    def get_always_skills(self) -> list[str]:
        """Get skills marked as always=true that meet requirements."""
        result = []
        for s in self.list_skills(filter_unavailable=True):
            meta = self.get_skill_metadata(s["name"]) or {}
            skill_meta = self._parse_exoclaw_metadata(meta.get("metadata", ""))
            if skill_meta.get("always") or meta.get("always"):
                result.append(s["name"])
        return result

    def get_skill_metadata(self, name: str) -> dict[str, Any] | None:
        """
        Get metadata from a skill's frontmatter.

        Args:
            name: Skill name.

        Returns:
            Metadata dict or None.
        """
        content = self.load_skill(name)
        if not content:
            return None

        if content.startswith("---"):
            match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
            if match:
                metadata = {}
                for line in match.group(1).split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip().strip("\"'")
                return metadata

        return None
