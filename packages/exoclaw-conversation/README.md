# exoclaw-conversation

File-backed conversation state manager implementing the exoclaw `Conversation` protocol.

## Install

```
pip install exoclaw-conversation
```

## Usage

```python
from pathlib import Path
from exoclaw_conversation.conversation import DefaultConversation

conversation = DefaultConversation.create(
    workspace=Path("~/.nanobot/workspace").expanduser(),
    provider=provider,   # any exoclaw LLMProvider
    model="anthropic/claude-opus-4-5",
)

messages = await conversation.build_prompt("session-1", "Hello!")
await conversation.record("session-1", new_messages)
```

`DefaultConversation.create()` wires the standard file-backed `SessionManager`, `MemoryStore`, and `ContextBuilder`. Each component can also be supplied independently via the constructor for custom setups.
