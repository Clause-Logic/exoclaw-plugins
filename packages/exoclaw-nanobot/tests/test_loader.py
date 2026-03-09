"""Tests for exoclaw_nanobot.config.loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exoclaw_nanobot.config.loader import (
    _migrate_config,
    get_config_path,
    load_config,
    save_config,
)
from exoclaw_nanobot.config.schema import Config


class TestGetConfigPath:
    def test_returns_nanobot_path(self) -> None:
        path = get_config_path()
        assert ".nanobot" in str(path)
        assert path.name == "config.json"


class TestLoadConfig:
    def test_returns_default_when_no_file(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path / "nonexistent.json")
        assert isinstance(cfg, Config)
        assert cfg.agents.defaults.model == "anthropic/claude-opus-4-5"

    def test_loads_valid_file(self, tmp_path: Path) -> None:
        p = tmp_path / "config.json"
        p.write_text(json.dumps({"agents": {"defaults": {"maxTokens": 1234}}}))
        cfg = load_config(p)
        assert cfg.agents.defaults.max_tokens == 1234

    def test_returns_default_on_invalid_json(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        p = tmp_path / "config.json"
        p.write_text("not json!!!")
        cfg = load_config(p)
        assert isinstance(cfg, Config)
        out = capsys.readouterr().out
        assert "Warning" in out

    def test_returns_default_on_validation_error(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        p = tmp_path / "config.json"
        # agents.defaults.maxTokens should be int, not a dict
        p.write_text(json.dumps({"agents": {"defaults": {"maxTokens": {"bad": "value"}}}}))
        cfg = load_config(p)
        assert isinstance(cfg, Config)
        out = capsys.readouterr().out
        assert "Warning" in out

    def test_runs_migration(self, tmp_path: Path) -> None:
        p = tmp_path / "config.json"
        # restrictToWorkspace only in exec, not yet at top-level — migration should promote it
        data = {"tools": {"exec": {"restrictToWorkspace": True}}}
        p.write_text(json.dumps(data))
        cfg = load_config(p)
        assert cfg.tools.restrict_to_workspace is True


class TestSaveConfig:
    def test_saves_and_loads_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "config.json"
        cfg = Config()
        cfg.agents.defaults.max_tokens = 9999
        save_config(cfg, p)
        loaded = load_config(p)
        assert loaded.agents.defaults.max_tokens == 9999

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "deep" / "nested" / "config.json"
        save_config(Config(), p)
        assert p.exists()

    def test_file_is_valid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "config.json"
        save_config(Config(), p)
        data = json.loads(p.read_text())
        assert isinstance(data, dict)
        assert "agents" in data


class TestMigrateConfig:
    def test_migrates_restrict_to_workspace(self) -> None:
        data: dict[str, object] = {"tools": {"exec": {"restrictToWorkspace": True}}}
        result = _migrate_config(data)
        tools = result["tools"]
        assert isinstance(tools, dict)
        assert tools.get("restrictToWorkspace") is True
        exec_cfg = tools.get("exec", {})
        assert isinstance(exec_cfg, dict)
        assert "restrictToWorkspace" not in exec_cfg

    def test_no_migration_needed(self) -> None:
        data: dict[str, object] = {"tools": {"exec": {}, "restrictToWorkspace": True}}
        result = _migrate_config(data)
        tools = result["tools"]
        assert isinstance(tools, dict)
        assert tools.get("restrictToWorkspace") is True

    def test_missing_tools_key(self) -> None:
        data: dict[str, object] = {}
        result = _migrate_config(data)
        assert result == {}

    def test_non_dict_tools_ignored(self) -> None:
        data: dict[str, object] = {"tools": "not_a_dict"}
        result = _migrate_config(data)
        assert result["tools"] == "not_a_dict"

    def test_non_dict_exec_ignored(self) -> None:
        data: dict[str, object] = {"tools": {"exec": "not_a_dict"}}
        result = _migrate_config(data)
        tools = result["tools"]
        assert isinstance(tools, dict)
        assert tools.get("exec") == "not_a_dict"
