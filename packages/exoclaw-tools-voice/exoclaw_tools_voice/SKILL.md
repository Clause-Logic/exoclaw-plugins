# voice

When the user wants to talk instead of type — or asks you to listen — call the
`listen` tool. It captures audio from the device microphone and returns the
spoken words as plain text.

## Usage

```
listen()
```

No parameters. Recording starts immediately and stops automatically after a
short silence or when the maximum duration is reached.

## Result shape

The tool returns the transcribed text (one or two sentences for typical
voice commands). Treat the result as if the user had typed it — proceed
with whatever action they asked for.

## Examples

- User says "set my display to a picture of a cat" → tool returns
  `"Set my display to a picture of a cat."` → you proceed by web-searching,
  fetching, and calling `repaint_screen`.
- User says "what's the weather" → tool returns `"What's the weather?"` →
  you proceed with `web_search`.

If the tool returns `(no speech detected)` the user pressed the talk
trigger but didn't actually say anything. Acknowledge briefly and wait
for a follow-up.
