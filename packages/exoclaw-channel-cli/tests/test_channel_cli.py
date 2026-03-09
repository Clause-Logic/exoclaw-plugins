"""Tests for exoclaw-channel-cli package."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exoclaw_channel_cli.channel import (
    CLIChannel,
    EXIT_COMMANDS,
    _flush_pending_tty_input,
    _print_response,
    _restore_terminal,
)
from exoclaw.bus.events import OutboundMessage
from exoclaw.bus.queue import MessageBus


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

class TestFlushPendingTtyInput:
    def test_non_tty_returns(self) -> None:
        _flush_pending_tty_input()

    def test_stdin_no_fileno(self) -> None:
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.fileno.side_effect = Exception("no fd")
            _flush_pending_tty_input()

    def test_termios_exception_handled(self) -> None:
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.fileno.return_value = 0
            with patch("os.isatty", return_value=True):
                with patch("termios.tcflush", side_effect=Exception("no tty")):
                    with patch("select.select", return_value=([], [], [])):
                        _flush_pending_tty_input()


class TestRestoreTerminal:
    def test_no_saved_attrs(self) -> None:
        import exoclaw_channel_cli.channel as ch
        ch._SAVED_TERM_ATTRS = None
        _restore_terminal()

    def test_with_saved_attrs(self) -> None:
        import exoclaw_channel_cli.channel as ch
        ch._SAVED_TERM_ATTRS = [1, 2, 3]
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.fileno.return_value = 0
            with patch("termios.tcsetattr") as mock_set:
                _restore_terminal()
                mock_set.assert_called_once()
        ch._SAVED_TERM_ATTRS = None

    def test_termios_exception_handled(self) -> None:
        import exoclaw_channel_cli.channel as ch
        ch._SAVED_TERM_ATTRS = [1, 2, 3]
        with patch("termios.tcsetattr", side_effect=Exception("no tty")):
            _restore_terminal()
        ch._SAVED_TERM_ATTRS = None


class TestPrintResponse:
    def test_renders_without_error(self) -> None:
        _print_response("Hello **world**", render_markdown=True)

    def test_plain_text(self) -> None:
        _print_response("plain text", render_markdown=False)

    def test_empty_content(self) -> None:
        _print_response("", render_markdown=True)


# ---------------------------------------------------------------------------
# CLIChannel
# ---------------------------------------------------------------------------

class TestCLIChannel:
    def test_name(self) -> None:
        assert CLIChannel().name == "cli"

    async def test_stop_sets_not_running(self) -> None:
        ch = CLIChannel()
        ch._running = True
        await ch.stop()
        assert not ch._running

    async def test_send_calls_print(self) -> None:
        ch = CLIChannel()
        msg = OutboundMessage(channel="cli", chat_id="direct", content="hello")
        with patch("exoclaw_channel_cli.channel._print_response") as mock_print:
            await ch.send(msg)
            mock_print.assert_called_once_with("hello", ch._render_markdown)

    async def test_start_keyboard_interrupt(self) -> None:
        ch = CLIChannel()
        bus = MessageBus()
        with patch("exoclaw_channel_cli.channel._init_prompt_session"):
            with patch("exoclaw_channel_cli.channel._read_interactive_input_async",
                       side_effect=KeyboardInterrupt):
                with patch("exoclaw_channel_cli.channel._restore_terminal"):
                    await ch.start(bus)
        assert not ch._running

    async def test_start_eof_exits(self) -> None:
        ch = CLIChannel()
        bus = MessageBus()
        with patch("exoclaw_channel_cli.channel._init_prompt_session"):
            with patch("exoclaw_channel_cli.channel._read_interactive_input_async",
                       side_effect=EOFError):
                with patch("exoclaw_channel_cli.channel._restore_terminal"):
                    await ch.start(bus)
        assert not ch._running

    async def test_start_exit_command(self) -> None:
        ch = CLIChannel()
        bus = MessageBus()
        with patch("exoclaw_channel_cli.channel._init_prompt_session"):
            with patch("exoclaw_channel_cli.channel._read_interactive_input_async",
                       return_value="exit"):
                with patch("exoclaw_channel_cli.channel._flush_pending_tty_input"):
                    with patch("exoclaw_channel_cli.channel._restore_terminal"):
                        await ch.start(bus)
        assert not ch._running

    async def test_start_empty_input_skipped(self) -> None:
        ch = CLIChannel()
        bus = MessageBus()
        inputs = iter(["", "  ", "exit"])

        async def _fake_input() -> str:
            return next(inputs)

        with patch("exoclaw_channel_cli.channel._init_prompt_session"):
            with patch("exoclaw_channel_cli.channel._read_interactive_input_async",
                       side_effect=_fake_input):
                with patch("exoclaw_channel_cli.channel._flush_pending_tty_input"):
                    with patch("exoclaw_channel_cli.channel._restore_terminal"):
                        await ch.start(bus)
        assert bus.inbound.empty()

    async def test_start_publishes_inbound_message(self) -> None:
        """User input gets published to the bus inbound queue."""
        ch = CLIChannel(chat_id="test")
        bus = MessageBus()
        inputs = iter(["hello world", "exit"])

        async def _fake_input() -> str:
            return next(inputs)

        # Make turn_done get set immediately so the wait() doesn't block
        async def _instant_outbound() -> OutboundMessage:
            await asyncio.sleep(0)
            return OutboundMessage(channel="cli", chat_id="test", content="response")

        with patch("exoclaw_channel_cli.channel._init_prompt_session"):
            with patch("exoclaw_channel_cli.channel._read_interactive_input_async",
                       side_effect=_fake_input):
                with patch("exoclaw_channel_cli.channel._flush_pending_tty_input"):
                    with patch("exoclaw_channel_cli.channel._restore_terminal"):
                        with patch.object(bus, "consume_outbound",
                                          side_effect=_instant_outbound):
                            await ch.start(bus)

        assert not bus.inbound.empty()
        msg = bus.inbound.get_nowait()
        assert msg.content == "hello world"
        assert msg.channel == "cli"

    def test_default_chat_id(self) -> None:
        ch = CLIChannel()
        assert ch._chat_id == "direct"

    def test_render_markdown_default(self) -> None:
        ch = CLIChannel()
        assert ch._render_markdown is True


class TestExitCommands:
    def test_contains_expected_commands(self) -> None:
        assert "exit" in EXIT_COMMANDS
        assert "quit" in EXIT_COMMANDS
        assert "/exit" in EXIT_COMMANDS
        assert "/quit" in EXIT_COMMANDS
        assert ":q" in EXIT_COMMANDS


# ---------------------------------------------------------------------------
# Additional coverage: _flush_pending_tty_input, _init_prompt_session,
# _read_interactive_input_async
# ---------------------------------------------------------------------------

class TestFlushPendingTtyInputExtra:
    def test_isatty_true_termios_succeeds(self) -> None:
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.fileno.return_value = 0
            with patch("os.isatty", return_value=True):
                with patch("termios.tcflush") as mock_flush:
                    _flush_pending_tty_input()
                    mock_flush.assert_called_once()

    def test_isatty_false_returns_early(self) -> None:
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.fileno.return_value = 0
            with patch("os.isatty", return_value=False):
                _flush_pending_tty_input()  # should return early without flushing

    def test_select_fallback_empty_bytes(self) -> None:
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.fileno.return_value = 0
            with patch("os.isatty", return_value=True):
                with patch("termios.tcflush", side_effect=ImportError):
                    # select returns ready once, then empty to break loop
                    with patch("select.select", side_effect=[([], [], [])]):
                        _flush_pending_tty_input()

    def test_select_fallback_read_returns_empty(self) -> None:
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.fileno.return_value = 0
            with patch("os.isatty", return_value=True):
                with patch("termios.tcflush", side_effect=ImportError):
                    with patch("select.select", side_effect=[([0], [], []), ([], [], [])]):
                        with patch("os.read", return_value=b""):
                            _flush_pending_tty_input()


class TestInitPromptSession:
    def test_creates_session(self) -> None:
        from exoclaw_channel_cli.channel import _init_prompt_session
        import exoclaw_channel_cli.channel as ch
        with patch("termios.tcgetattr", return_value=[1, 2, 3]):
            _init_prompt_session()
        assert ch._PROMPT_SESSION is not None
        ch._PROMPT_SESSION = None
        ch._SAVED_TERM_ATTRS = None

    def test_termios_exception_handled(self) -> None:
        from exoclaw_channel_cli.channel import _init_prompt_session
        import exoclaw_channel_cli.channel as ch
        with patch("termios.tcgetattr", side_effect=Exception("no tty")):
            _init_prompt_session()
        ch._PROMPT_SESSION = None
        ch._SAVED_TERM_ATTRS = None


class TestReadInteractiveInputAsync:
    async def test_raises_without_session(self) -> None:
        from exoclaw_channel_cli.channel import _read_interactive_input_async
        import exoclaw_channel_cli.channel as ch
        ch._PROMPT_SESSION = None
        with pytest.raises(RuntimeError):
            await _read_interactive_input_async()

    async def test_eof_becomes_keyboard_interrupt(self) -> None:
        from exoclaw_channel_cli.channel import _read_interactive_input_async
        import exoclaw_channel_cli.channel as ch
        mock_session = MagicMock()
        mock_session.prompt_async = AsyncMock(side_effect=EOFError)
        ch._PROMPT_SESSION = mock_session
        with pytest.raises(KeyboardInterrupt):
            await _read_interactive_input_async()
        ch._PROMPT_SESSION = None
