"""Tests for SkillsLoader's ``allowed_names`` deployment-scoped allow-list.

When a deployment ships several skills in the builtin/workspace/package
trees but only wants the model to see a subset, ``allowed_names`` makes
the loader behave as if disallowed skills don't exist — they're absent
from ``list_skills``, ``build_skills_summary``, ``load_skill``,
``activate_skill``, hook discovery, and ``get_always_skills``.
"""

from __future__ import annotations

from pathlib import Path

from exoclaw_conversation.skills import SkillsLoader


def _write_skill(root: Path, name: str, body: str = "skill content", **frontmatter: str) -> None:
    """Drop a SKILL.md at ``root/name/SKILL.md`` with optional frontmatter."""
    (root / name).mkdir(parents=True, exist_ok=True)
    fm_lines = "\n".join(f"{k}: {v}" for k, v in frontmatter.items())
    fm_block = f"---\n{fm_lines}\n---\n" if fm_lines else ""
    (root / name / "SKILL.md").write_text(f"{fm_block}{body}")


class TestAllowListListSkills:
    def test_no_allow_list_lists_everything(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        _write_skill(builtin, "alpha", description="A")
        _write_skill(builtin, "beta", description="B")

        loader = SkillsLoader(workspace=tmp_path / "ws", builtin_skills_dir=builtin)
        names = {s["name"] for s in loader.list_skills()}
        assert names == {"alpha", "beta"}

    def test_allow_list_filters_builtin(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        _write_skill(builtin, "alpha", description="A")
        _write_skill(builtin, "beta", description="B")
        _write_skill(builtin, "gamma", description="C")

        loader = SkillsLoader(
            workspace=tmp_path / "ws",
            builtin_skills_dir=builtin,
            allowed_names=["alpha", "gamma"],
        )
        names = {s["name"] for s in loader.list_skills()}
        assert names == {"alpha", "gamma"}

    def test_allow_list_filters_workspace(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _write_skill(ws / "skills", "alpha", description="A")
        _write_skill(ws / "skills", "beta", description="B")

        loader = SkillsLoader(workspace=ws, allowed_names=["alpha"])
        names = {s["name"] for s in loader.list_skills()}
        assert names == {"alpha"}

    def test_empty_allow_list_hides_everything(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        _write_skill(builtin, "alpha", description="A")

        loader = SkillsLoader(
            workspace=tmp_path / "ws",
            builtin_skills_dir=builtin,
            allowed_names=[],
        )
        assert loader.list_skills() == []


class TestAllowListLoadSkill:
    def test_disallowed_load_returns_none(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        _write_skill(builtin, "secret", body="don't show me")

        loader = SkillsLoader(
            workspace=tmp_path / "ws",
            builtin_skills_dir=builtin,
            allowed_names=["other"],
        )
        assert loader.load_skill("secret") is None

    def test_allowed_load_returns_content(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        _write_skill(builtin, "ok", body="visible content")

        loader = SkillsLoader(
            workspace=tmp_path / "ws",
            builtin_skills_dir=builtin,
            allowed_names=["ok"],
        )
        content = loader.load_skill("ok")
        assert content is not None
        assert "visible content" in content


class TestAllowListActivateSkill:
    def test_disallowed_activate_returns_not_found(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        _write_skill(builtin, "secret", body="x")

        loader = SkillsLoader(
            workspace=tmp_path / "ws",
            builtin_skills_dir=builtin,
            allowed_names=[],
        )
        result = loader.activate_skill("secret")
        # The model gets the same "not found" message it'd get for a
        # genuinely missing skill — no signal that it's been hidden.
        assert "not found" in result.content.lower()
        assert result.tool_names == []


class TestAllowListSummary:
    def test_disallowed_skills_absent_from_summary(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        _write_skill(builtin, "shown", description="visible")
        _write_skill(builtin, "hidden", description="should not appear")

        loader = SkillsLoader(
            workspace=tmp_path / "ws",
            builtin_skills_dir=builtin,
            allowed_names=["shown"],
        )
        summary = loader.build_skills_summary()
        assert "<name>shown</name>" in summary
        assert "hidden" not in summary
