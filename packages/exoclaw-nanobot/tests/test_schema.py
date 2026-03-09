"""Tests for exoclaw_nanobot.config.schema."""

from __future__ import annotations

import pytest

from exoclaw_nanobot.config.schema import (
    AgentDefaults,
    Config,
    MCPServerConfig,
    ProviderConfig,
    ToolsConfig,
)


class TestConfigDefaults:
    def test_creates_with_defaults(self) -> None:
        cfg = Config()
        assert cfg.agents.defaults.model == "anthropic/claude-opus-4-5"
        assert cfg.agents.defaults.provider == "auto"
        assert cfg.agents.defaults.max_tokens == 8192

    def test_workspace_path_expanded(self) -> None:
        cfg = Config()
        path = cfg.workspace_path
        assert not str(path).startswith("~")
        assert ".exoclaw" in str(path)

    def test_custom_workspace(self) -> None:
        cfg = Config()
        cfg.agents.defaults.workspace = "/tmp/test_workspace"
        assert str(cfg.workspace_path) == "/tmp/test_workspace"

    def test_env_prefix_is_exoclaw(self) -> None:
        assert cfg_env_prefix() == "EXOCLAW_"

    def test_channels_defaults(self) -> None:
        cfg = Config()
        assert cfg.channels.send_progress is True
        assert cfg.channels.telegram.enabled is False
        assert cfg.channels.discord.enabled is False

    def test_gateway_heartbeat_defaults(self) -> None:
        cfg = Config()
        assert cfg.gateway.heartbeat.enabled is True
        assert cfg.gateway.heartbeat.interval_s == 1800

    def test_tools_defaults(self) -> None:
        cfg = Config()
        assert cfg.tools.restrict_to_workspace is False
        assert cfg.tools.exec.timeout == 10
        assert cfg.tools.web.search.max_results == 5


def cfg_env_prefix() -> str:
    from pydantic_settings import BaseSettings
    cfg = Config()
    return cfg.model_config.get("env_prefix", "")


class TestMatchProvider:
    def _cfg_with_key(self, provider: str, key: str) -> Config:
        cfg = Config()
        getattr(cfg.providers, provider).api_key = key
        return cfg

    def test_auto_matches_anthropic_by_model(self) -> None:
        cfg = self._cfg_with_key("anthropic", "sk-ant-test")
        p, name = cfg._match_provider("claude-opus-4-5")
        assert name == "anthropic"
        assert p is not None
        assert p.api_key == "sk-ant-test"

    def test_auto_matches_openai_by_model(self) -> None:
        cfg = self._cfg_with_key("openai", "sk-openai-test")
        p, name = cfg._match_provider("gpt-4o")
        assert name == "openai"

    def test_forced_provider(self) -> None:
        cfg = Config()
        cfg.agents.defaults.provider = "deepseek"
        cfg.providers.deepseek.api_key = "ds-key"
        p, name = cfg._match_provider()
        assert name == "deepseek"

    def test_forced_invalid_provider_returns_none(self) -> None:
        cfg = Config()
        cfg.agents.defaults.provider = "nonexistent_xyz"
        p, name = cfg._match_provider()
        assert p is None
        assert name is None

    def test_no_key_falls_through(self) -> None:
        cfg = Config()  # no keys set
        p, name = cfg._match_provider("claude-opus-4-5")
        assert p is None
        assert name is None

    def test_fallback_uses_first_available_key(self) -> None:
        cfg = Config()
        cfg.providers.groq.api_key = "groq-key"
        p, name = cfg._match_provider("totally-unknown-model")
        assert name == "groq"

    def test_get_api_key(self) -> None:
        cfg = self._cfg_with_key("anthropic", "sk-ant-key")
        assert cfg.get_api_key("claude-opus-4-5") == "sk-ant-key"

    def test_get_api_key_none_when_no_match(self) -> None:
        cfg = Config()
        assert cfg.get_api_key("claude-opus-4-5") is None

    def test_get_provider_name(self) -> None:
        cfg = self._cfg_with_key("deepseek", "ds-key")
        assert cfg.get_provider_name("deepseek-chat") == "deepseek"

    def test_get_api_base_from_provider_config(self) -> None:
        cfg = self._cfg_with_key("anthropic", "sk-ant")
        cfg.providers.anthropic.api_base = "https://custom.base/v1"
        assert cfg.get_api_base("claude-opus-4-5") == "https://custom.base/v1"

    def test_get_api_base_gateway_default(self) -> None:
        cfg = self._cfg_with_key("openrouter", "sk-or-key")
        base = cfg.get_api_base("openrouter/claude-3")
        assert base == "https://openrouter.ai/api/v1"

    def test_get_api_base_none_for_standard_provider(self) -> None:
        cfg = self._cfg_with_key("anthropic", "sk-ant")
        base = cfg.get_api_base("claude-opus-4-5")
        assert base is None


class TestMCPServerConfig:
    def test_defaults(self) -> None:
        srv = MCPServerConfig()
        assert srv.type is None
        assert srv.command == ""
        assert srv.tool_timeout == 30

    def test_stdio(self) -> None:
        srv = MCPServerConfig(command="npx", args=["-y", "server"])
        assert srv.command == "npx"
        assert srv.args == ["-y", "server"]

    def test_sse(self) -> None:
        srv = MCPServerConfig(url="http://localhost/sse")
        assert srv.url == "http://localhost/sse"


class TestCamelCaseAlias:
    def test_snake_and_camel_both_work(self) -> None:
        from exoclaw_nanobot.config.schema import AgentsConfig

        via_snake = AgentsConfig.model_validate({"defaults": {"max_tokens": 999}})
        via_camel = AgentsConfig.model_validate({"defaults": {"maxTokens": 999}})
        assert via_snake.defaults.max_tokens == 999
        assert via_camel.defaults.max_tokens == 999
