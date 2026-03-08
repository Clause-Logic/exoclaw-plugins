"""Test DefaultConversation.record() — runtime context stripping and image placeholder logic."""

from unittest.mock import MagicMock

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.conversation import DefaultConversation


def _mk_conv(tmp_path) -> DefaultConversation:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return DefaultConversation(workspace=tmp_path, provider=provider, model="test-model")


@pytest.mark.asyncio
async def test_record_skips_multimodal_user_when_only_runtime_context(tmp_path) -> None:
    conv = _mk_conv(tmp_path)
    runtime = ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now (UTC)"

    await conv.record(
        "test:runtime-only",
        [{"role": "user", "content": [{"type": "text", "text": runtime}]}],
    )

    session = conv._sessions.get_or_create("test:runtime-only")
    assert session.messages == []


@pytest.mark.asyncio
async def test_record_keeps_image_placeholder_after_runtime_strip(tmp_path) -> None:
    conv = _mk_conv(tmp_path)
    runtime = ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now (UTC)"

    await conv.record(
        "test:image",
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": runtime},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }],
    )

    session = conv._sessions.get_or_create("test:image")
    assert session.messages[0]["content"] == [{"type": "text", "text": "[image]"}]
