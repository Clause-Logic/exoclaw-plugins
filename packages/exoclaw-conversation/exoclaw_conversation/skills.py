"""Skills loader for agent capabilities."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

# Default builtin skills directory — none bundled in this package
BUILTIN_SKILLS_DIR: Path | None = None


class SkillsLoader:
    """
    Loader for agent skills.

    Skills are markdown files (SKILL.md) that teach the agent how to use
    specific tools or perform certain tasks.
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
        if skill_packages:
            self._load_package_skills(skill_packages)

    def _load_package_skills(self, packages: list[str]) -> None:
        """Load skills from installed Python packages via entry points."""
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
                except Exception:
                    pass

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
                        skills.append({"name": skill_dir.name, "path": str(skill_file), "source": "workspace"})

        # Built-in skills
        if self.builtin_skills and self.builtin_skills.exists():
            for skill_dir in self.builtin_skills.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists() and not any(s["name"] == skill_dir.name for s in skills):
                        skills.append({"name": skill_dir.name, "path": str(skill_file), "source": "builtin"})

        # Package skills (lowest priority — workspace and builtin override)
        for name in self._package_skills:
            if not any(s["name"] == name for s in skills):
                skills.append({"name": name, "path": f"package:{name}", "source": "package"})

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

    def build_skills_summary(self) -> str:
        """
        Build a summary of all skills (name, description, path, availability).

        Returns:
            XML-formatted skills summary.
        """
        all_skills = self.list_skills(filter_unavailable=False)
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

            lines.append(f"  <skill available=\"{str(available).lower()}\">")
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
            if not shutil.which(b):
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
                return content[match.end():].strip()
        return content

    def _parse_exoclaw_metadata(self, raw: str) -> dict[str, Any]:
        """Parse skill metadata JSON from frontmatter (supports exoclaw, nanobot, openclaw keys)."""
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                result = data.get("exoclaw", data.get("nanobot", data.get("openclaw", {})))
                return dict(result) if isinstance(result, dict) else {}
            return {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def _check_requirements(self, skill_meta: dict[str, Any]) -> bool:
        """Check if skill requirements are met (bins, env vars)."""
        requires = skill_meta.get("requires", {})
        for b in requires.get("bins", []):
            if not shutil.which(b):
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
        for skills_dir in (self.workspace_skills, self.builtin_skills):
            if not skills_dir or not skills_dir.exists():
                continue
            for skill_dir in skills_dir.iterdir():
                if not skill_dir.is_dir():
                    continue
                # Support both exoclaw and legacy nanobot hook paths
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
        for skills_dir in (self.workspace_skills, self.builtin_skills):
            if not skills_dir or not skills_dir.exists():
                continue
            for skill_dir in sorted(skills_dir.iterdir()):
                if not skill_dir.is_dir():
                    continue
                # Support both exoclaw and legacy nanobot hook paths
                for hook_file in (
                    skill_dir / "hooks" / "exoclaw" / hook_name,
                    skill_dir / "hooks" / "nanobot" / hook_name,
                ):
                    if hook_file.exists() and os.access(hook_file, os.X_OK):
                        results.append(hook_file)
                        break
        return results

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
                        metadata[key.strip()] = value.strip().strip('"\'')
                return metadata

        return None
