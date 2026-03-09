# exoclaw-tools-message

Message sending tool implementing the exoclaw `ToolBase` protocol — lets the agent deliver messages to users on any channel the bus supports.

## Install

```
pip install exoclaw-tools-message
```

## Usage

```python
from exoclaw_tools_message.tool import MessageTool

message_tool = MessageTool(
    send_callback=bus.publish_outbound,
    default_channel="cli",
    default_chat_id="direct",
    suppress_patterns=[r"^HEARTBEAT_OK$"],   # optional: filter unwanted lines
)

# Update context per turn so the tool knows where to deliver replies
message_tool.set_context(channel="cli", chat_id="direct")
```

`MessageTool` is the mechanism by which the agent sends outbound replies. It tracks whether a message was sent in the current turn and supports per-message media attachments.
