"""Tests for exoclaw_nanobot.providers."""

from __future__ import annotations

import pytest

from exoclaw_nanobot.providers import (
    PROVIDERS,
    ProviderSpec,
    find_by_model,
    find_by_name,
    find_gateway,
)


class TestProviderSpec:
    def test_label_uses_display_name(self) -> None:
        spec = ProviderSpec(name="foo", keywords=(), env_key="", display_name="Foo Bar")
        assert spec.label == "Foo Bar"

    def test_label_falls_back_to_title(self) -> None:
        spec = ProviderSpec(name="deepseek", keywords=(), env_key="")
        assert spec.label == "Deepseek"

    def test_frozen(self) -> None:
        spec = ProviderSpec(name="x", keywords=(), env_key="")
        with pytest.raises(Exception):
            spec.name = "y"  # type: ignore[misc]


class TestFindByName:
    def test_finds_anthropic(self) -> None:
        spec = find_by_name("anthropic")
        assert spec is not None
        assert spec.name == "anthropic"

    def test_finds_deepseek(self) -> None:
        spec = find_by_name("deepseek")
        assert spec is not None
        assert spec.litellm_prefix == "deepseek"

    def test_returns_none_for_unknown(self) -> None:
        assert find_by_name("nonexistent_provider_xyz") is None

    def test_all_providers_findable(self) -> None:
        for spec in PROVIDERS:
            found = find_by_name(spec.name)
            assert found is spec


class TestFindByModel:
    def test_claude_matches_anthropic(self) -> None:
        spec = find_by_model("claude-opus-4-5")
        assert spec is not None
        assert spec.name == "anthropic"

    def test_gpt_matches_openai(self) -> None:
        spec = find_by_model("gpt-4o")
        assert spec is not None
        assert spec.name == "openai"

    def test_deepseek_by_keyword(self) -> None:
        spec = find_by_model("deepseek-chat")
        assert spec is not None
        assert spec.name == "deepseek"

    def test_gemini_by_keyword(self) -> None:
        spec = find_by_model("gemini-pro")
        assert spec is not None
        assert spec.name == "gemini"

    def test_anthropic_prefix_wins(self) -> None:
        spec = find_by_model("anthropic/claude-opus-4-5")
        assert spec is not None
        assert spec.name == "anthropic"

    def test_kimi_matches_moonshot(self) -> None:
        spec = find_by_model("kimi-k2.5")
        assert spec is not None
        assert spec.name == "moonshot"

    def test_qwen_matches_dashscope(self) -> None:
        spec = find_by_model("qwen-max")
        assert spec is not None
        assert spec.name == "dashscope"

    def test_unknown_returns_none(self) -> None:
        assert find_by_model("totally-unknown-model-xyz") is None

    def test_groq_keyword(self) -> None:
        spec = find_by_model("groq/llama3-8b")
        assert spec is not None
        assert spec.name == "groq"

    def test_glm_matches_zhipu(self) -> None:
        spec = find_by_model("glm-4")
        assert spec is not None
        assert spec.name == "zhipu"


class TestFindGateway:
    def test_openrouter_by_key_prefix(self) -> None:
        spec = find_gateway(api_key="sk-or-abcdef")
        assert spec is not None
        assert spec.name == "openrouter"

    def test_aihubmix_by_base_keyword(self) -> None:
        spec = find_gateway(api_base="https://aihubmix.com/v1")
        assert spec is not None
        assert spec.name == "aihubmix"

    def test_by_provider_name(self) -> None:
        spec = find_gateway(provider_name="openrouter")
        assert spec is not None
        assert spec.name == "openrouter"

    def test_non_gateway_by_name_returns_none(self) -> None:
        # anthropic is not a gateway
        spec = find_gateway(provider_name="anthropic")
        assert spec is None

    def test_no_match_returns_none(self) -> None:
        assert find_gateway() is None

    def test_siliconflow_by_base(self) -> None:
        spec = find_gateway(api_base="https://api.siliconflow.cn/v1")
        assert spec is not None
        assert spec.name == "siliconflow"

    def test_vllm_by_name(self) -> None:
        spec = find_gateway(provider_name="vllm")
        assert spec is not None
        assert spec.name == "vllm"
        assert spec.is_local is True
