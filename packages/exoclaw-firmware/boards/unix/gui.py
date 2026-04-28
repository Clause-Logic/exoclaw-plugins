"""Unix sim GUI — pygame window + spacebar push-to-talk + HUD.

Spawns the MP sim subprocess, shows screen.png in a window,
overlays status pip + caption bar by parsing sim stdout.
Spacebar sends /talk to the sim.
"""

from __future__ import annotations

import os
import select
import subprocess
import sys
import textwrap
import time

import pygame


def _find_micropython_bin() -> str:
    return os.environ.get("EXOCLAW_MICROPYTHON_BIN", "micropython")


def _find_host_python() -> str:
    if "EXOCLAW_HOST_PYTHON" in os.environ:
        return os.environ["EXOCLAW_HOST_PYTHON"]
    try:
        result = subprocess.run(
            ["uv", "run", "python", "-c", "import sys; print(sys.executable)"],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    return "python3"


_WHITE = (255, 255, 255)
_DARK = (30, 30, 30)
_GREEN = (0, 200, 80)
_AMBER = (220, 160, 0)
_RED = (200, 60, 60)
_GRAY = (140, 140, 140)

_STATUS_COLORS = {
    "ready": _GREEN,
    "listening": _RED,
    "transcribing": _AMBER,
    "thinking": _AMBER,
}


def main() -> int:
    firmware_dir = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(firmware_dir, "mise.toml")):
        parent = os.path.dirname(firmware_dir)
        if parent == firmware_dir:
            return 1
        firmware_dir = parent

    stage_dir = os.path.join(firmware_dir, ".stage")
    screen_png = os.path.join(stage_dir, "screen.png")

    if not os.path.isdir(stage_dir):
        subprocess.run(
            ["mise", "run", "stage"], cwd=firmware_dir,
            env={**os.environ, "EXOCLAW_BOARD": "unix"}, check=True,
        )

    host_python = _find_host_python()
    mp_bin = _find_micropython_bin()

    sim_env = {
        **os.environ,
        "EXOCLAW_HOST_PYTHON": host_python,
        "EXOCLAW_SCREEN_OUT": screen_png,
        "MICROPYPATH": ".frozen:" + stage_dir,
    }
    sim_env.pop("PYTHONPATH", None)

    sim_proc = subprocess.Popen(
        [mp_bin, "main.py"], cwd=stage_dir,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, bufsize=0,
    )

    pygame.init()
    WIDTH, HEIGHT = 800, 480
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("exoclaw sim")
    clock = pygame.time.Clock()

    window.fill(_WHITE)
    pygame.display.flip()

    try:
        pip_font = pygame.font.SysFont("Helvetica", 11)
        caption_font = pygame.font.SysFont("Helvetica", 13)
    except Exception:
        pip_font = pygame.font.SysFont(None, 14)
        caption_font = pygame.font.SysFont(None, 16)

    last_mtime: float = 0.0
    panel_surface = pygame.Surface((WIDTH, HEIGHT))
    panel_surface.fill(_WHITE)
    status = "ready"
    caption = "spacebar to talk"
    caption_time: float = 0.0
    stdout_buf = b""

    def _process_line(raw: str) -> None:
        nonlocal status, caption, caption_time
        line = raw.strip()

        if "[listening...]" in line:
            status = "listening"
            caption = "listening..."
            caption_time = time.monotonic()
        elif "CAPTURED" in line:
            status = "transcribing"
            caption = "transcribing..."
            caption_time = time.monotonic()
        elif "SILENCE" in line and status == "listening":
            status = "ready"
            caption = "(no speech detected)"
            caption_time = time.monotonic()
        elif '"event": "turn_start"' in line:
            if status != "transcribing":
                status = "thinking"
            caption = "thinking..."
            caption_time = time.monotonic()
        elif '"event": "tool_call"' in line:
            status = "thinking"
            try:
                import json
                obj = json.loads(line)
                name = obj.get("tool.name", "")
                if name:
                    caption = "→ {}".format(name)
                    caption_time = time.monotonic()
            except Exception:
                pass
        elif "bot>" in line:
            idx = line.find("bot>")
            if idx >= 0:
                msg = line[idx + 4:].strip()
                if msg:
                    caption = msg
                    caption_time = time.monotonic()
            status = "ready"
        elif '"event": "turn_end"' in line:
            status = "ready"

        sys.stdout.write(raw + "\n")
        sys.stdout.flush()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if sim_proc.stdin and status == "ready":
                        status = "listening"
                        caption = "press spacebar... listening"
                        caption_time = time.monotonic()
                        try:
                            sim_proc.stdin.write(b"/talk\n")
                            sim_proc.stdin.flush()
                        except (BrokenPipeError, OSError):
                            pass

        # Drain sim stdout.
        if sim_proc.stdout:
            try:
                rlist, _, _ = select.select([sim_proc.stdout], [], [], 0)
                if rlist:
                    data = os.read(sim_proc.stdout.fileno(), 8192)
                    if data:
                        stdout_buf += data
                        while b"\n" in stdout_buf:
                            line_bytes, stdout_buf = stdout_buf.split(b"\n", 1)
                            _process_line(line_bytes.decode("utf-8", errors="replace"))
            except (OSError, ValueError):
                pass

        # Refresh panel from screen.png.
        try:
            mtime = os.path.getmtime(screen_png)
            if mtime > last_mtime:
                last_mtime = mtime
                time.sleep(0.05)
                try:
                    panel_surface = pygame.image.load(screen_png)
                    caption = ""
                except pygame.error:
                    pass
        except FileNotFoundError:
            pass

        # Compose: panel + HUD overlay.
        window.blit(panel_surface, (0, 0))

        # Status pip — top right.
        pip_color = _STATUS_COLORS.get(status, _GREEN)
        dot_x, dot_y, dot_r = WIDTH - 16, 14, 5
        pygame.draw.circle(window, pip_color, (dot_x, dot_y), dot_r)
        pip_label = pip_font.render(status, True, _GRAY)
        window.blit(pip_label, (dot_x - dot_r - pip_label.get_width() - 4, dot_y - 6))

        # Caption bar — bottom, fades after 15s.
        if caption and (time.monotonic() - caption_time) < 5.0:
            bar_h = 44
            bar_y = HEIGHT - bar_h
            bar_surface = pygame.Surface((WIDTH, bar_h), pygame.SRCALPHA)
            bar_surface.fill((30, 30, 30, 210))
            window.blit(bar_surface, (0, bar_y))
            lines = textwrap.wrap(caption, width=95)[:2]
            for i, ln in enumerate(lines):
                label = caption_font.render(ln, True, (240, 240, 240))
                window.blit(label, (10, bar_y + 6 + i * 18))

        pygame.display.flip()

        if sim_proc.poll() is not None:
            caption = "sim exited (rc={})".format(sim_proc.returncode)
            caption_time = time.monotonic()

        clock.tick(20)

    if sim_proc.poll() is None:
        sim_proc.terminate()
        try:
            sim_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            sim_proc.kill()
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
