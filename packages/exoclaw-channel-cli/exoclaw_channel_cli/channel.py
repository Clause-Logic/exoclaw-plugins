"""Interactive CLI REPL channel for exoclaw."""

from __future__ import annotations

import asyncio
import os
import select
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from exoclaw.bus.events import InboundMessage, OutboundMessage

if TYPE_CHECKING:
    from exoclaw.bus.protocol import Bus

console = Console()
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}

_PROMPT_SESSION: PromptSession | None = None
_SAVED_TERM_ATTRS = None


def _flush_pending_tty_input() -> None:
    """Drop unread keypresses typed while the model was generating output."""
    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return
    except Exception:
        return

    try:
        import termios
        termios.tcflush(fd, termios.TCIFLUSH)
        return
    except Exception:
        pass

    try:
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            if not os.read(fd, 4096):
                break
    except Exception:
        return


def _restore_terminal() -> None:
    """Restore terminal to its original state (echo, line buffering, etc.)."""
    if _SAVED_TERM_ATTRS is None:
        return
    try:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        pass


def _init_prompt_session(history_dir: Path | None = None) -> None:
    """Create the prompt_toolkit session with persistent file history."""
    global _PROMPT_SESSION, _SAVED_TERM_ATTRS

    try:
        import termios
        _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        pass

    hist_dir = history_dir or (Path.home() / ".exoclaw" / "history")
    hist_dir.mkdir(parents=True, exist_ok=True)
    history_file = hist_dir / "cli_history"

    _PROMPT_SESSION = PromptSession(
        history=FileHistory(str(history_file)),
        enable_open_in_editor=False,
        multiline=False,
    )


async def _read_interactive_input_async() -> str:
    """Read user input using prompt_toolkit."""
    if _PROMPT_SESSION is None:
        raise RuntimeError("Call _init_prompt_session() first")
    try:
        with patch_stdout():
            return await _PROMPT_SESSION.prompt_async(
                HTML("<b fg='ansiblue'>You:</b> "),
            )
    except EOFError as exc:
        raise KeyboardInterrupt from exc


def _print_response(response: str, render_markdown: bool = True) -> None:
    """Render assistant response with consistent terminal styling."""
    content = response or ""
    body = Markdown(content) if render_markdown else Text(content)
    console.print()
    console.print("[cyan]exoclaw[/cyan]")
    console.print(body)
    console.print()


class CLIChannel:
    """
    Interactive CLI REPL channel.

    Implements the exoclaw Channel protocol without inheriting from any
    exoclaw class.
    """

    name = "cli"

    def __init__(
        self,
        chat_id: str = "direct",
        render_markdown: bool = True,
        history_dir: Path | None = None,
    ):
        self._chat_id = chat_id
        self._render_markdown = render_markdown
        self._history_dir = history_dir
        self._running = False
        self._bus: Bus | None = None

    async def start(self, bus: Bus) -> None:
        """Connect and begin the interactive REPL loop."""
        self._bus = bus
        self._running = True

        _init_prompt_session(self._history_dir)
        console.print("exoclaw interactive mode (type [bold]exit[/bold] or [bold]Ctrl+C[/bold] to quit)\n")

        def _handle_signal(signum: int, frame: object) -> None:
            sig_name = signal.Signals(signum).name
            _restore_terminal()
            console.print(f"\nReceived {sig_name}, goodbye!")
            self._running = False

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, _handle_signal)
        if hasattr(signal, "SIGPIPE"):
            signal.signal(signal.SIGPIPE, signal.SIG_IGN)

        turn_done = asyncio.Event()
        turn_done.set()
        turn_response: list[str] = []

        async def _consume_outbound() -> None:
            while self._running:
                try:
                    msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
                    if msg.metadata.get("_progress"):
                        console.print(f"  [dim]↳ {msg.content}[/dim]")
                    elif not turn_done.is_set():
                        if msg.content:
                            turn_response.append(msg.content)
                        turn_done.set()
                    elif msg.content:
                        console.print()
                        _print_response(msg.content, self._render_markdown)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break

        outbound_task = asyncio.create_task(_consume_outbound())

        try:
            while self._running:
                try:
                    _flush_pending_tty_input()
                    user_input = await _read_interactive_input_async()
                    command = user_input.strip()
                    if not command:
                        continue

                    if command.lower() in EXIT_COMMANDS:
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break

                    turn_done.clear()
                    turn_response.clear()

                    await bus.publish_inbound(InboundMessage(
                        channel=self.name,
                        sender_id="user",
                        chat_id=self._chat_id,
                        content=user_input,
                    ))

                    with console.status("[dim]thinking...[/dim]", spinner="dots"):
                        await turn_done.wait()

                    if turn_response:
                        _print_response(turn_response[0], self._render_markdown)

                except KeyboardInterrupt:
                    _restore_terminal()
                    console.print("\nGoodbye!")
                    break
                except EOFError:
                    _restore_terminal()
                    console.print("\nGoodbye!")
                    break
        finally:
            self._running = False
            outbound_task.cancel()
            await asyncio.gather(outbound_task, return_exceptions=True)

    async def stop(self) -> None:
        """Signal the REPL loop to exit."""
        self._running = False
        logger.info("CLIChannel stopping")

    async def send(self, msg: OutboundMessage) -> None:
        """Print an outbound message to stdout."""
        _print_response(msg.content, self._render_markdown)
