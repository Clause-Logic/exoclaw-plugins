"""Message tool for sending messages to users."""

import re
from typing import Any, Awaitable, Callable

from exoclaw.agent.tools.protocol import ToolBase
from exoclaw.bus.events import OutboundMessage


def filter_suppressed(content: str, patterns: list[str]) -> str | None:
    """Filter lines matching suppress_patterns.

    Each non-empty line is tested against every pattern (case-insensitive).
    Matching lines are removed. If no non-empty lines remain, returns None
    (suppress the whole message). Blank lines are kept only if surrounded by
    kept content.
    """
    lines = content.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        suppressed = False
        for pattern in patterns:
            try:
                if re.search(pattern, stripped, re.IGNORECASE):
                    suppressed = True
                    break
            except re.error:
                pass
        if not suppressed:
            kept.append(line)
    result = "\n".join(kept).strip()
    return result if result else None


class MessageTool(ToolBase):
    """Tool to send messages to users on chat channels."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        default_message_id: str | None = None,
        suppress_patterns: list[str] | None = None,
        retrieval_fn: Callable[[str], Awaitable[str | None]] | None = None,
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id
        self._default_message_id = default_message_id
        self._sent_in_turn: bool = False
        self._suppress_patterns: list[str] = suppress_patterns or []
        self._retrieval_fn = retrieval_fn

    def set_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Set the current message context."""
        self._default_channel = channel
        self._default_chat_id = chat_id
        self._default_message_id = message_id

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    def start_turn(self) -> None:
        """Reset per-turn send tracking."""
        self._sent_in_turn = False

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return "Send a message to the user. Use this when you want to communicate something."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send"
                },
                "channel": {
                    "type": "string",
                    "description": "Optional: target channel (telegram, discord, etc.)"
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional: target chat/user ID"
                },
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: list of file paths to attach (images, audio, documents)"
                }
            },
            "required": ["content"]
        }

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        message_id: str | None = None,
        media: list[str] | None = None,
        **kwargs: Any
    ) -> str:
        channel = channel or self._default_channel
        chat_id = chat_id or self._default_chat_id
        message_id = message_id or self._default_message_id

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"

        if not self._send_callback:
            return "Error: Message sending not configured"

        if self._suppress_patterns:
            original = content
            filtered = filter_suppressed(content, self._suppress_patterns)
            if filtered is None:
                if self._retrieval_fn:
                    learnings = await self._retrieval_fn(original)
                    if learnings:
                        return f"[Message suppressed, not sent to user]\n\n{learnings}"
                return "Message suppressed by suppress_patterns"
            content = filtered

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            media=media or [],
            metadata={
                "message_id": message_id,
            }
        )

        try:
            await self._send_callback(msg)
            if channel == self._default_channel and chat_id == self._default_chat_id:
                self._sent_in_turn = True
            media_info = f" with {len(media)} attachments" if media else ""
            return f"Message sent to {channel}:{chat_id}{media_info}"
        except Exception as e:
            return f"Error sending message: {str(e)}"
