"""ESP32-S3 boot script — runs once before ``main.py``.

Sets up the things the agent stack needs before it starts:
- Mount the SD card at ``/sd`` (workspace root for sessions, memory,
  skills). Skipped silently if no card is present so the board can
  still boot for diagnostics.
- Connect to WiFi using credentials from ``secrets.py`` (gitignored;
  see ``secrets.py.example`` for the template).
- Sync the RTC via NTP so TLS cert validation has a sane clock —
  certificate ``notBefore`` checks fail spectacularly when the chip
  thinks it's 1970.

This file lives in flash root (``/boot.py``) on the device. Copy
it there with ``mise run flash`` (see the package ``mise.toml``).
"""

import time

import machine
import network


def _mount_sd() -> bool:
    """Mount an SPI-attached SD card at ``/sd``. Returns True on
    success, False if no card was found / mount failed.

    Pin assignments below match the common ESP32-S3 dev-board
    layout (sclk=12, mosi=11, miso=13, cs=10). Override in your
    board variant if your wiring differs."""
    try:
        import os

        sd = machine.SDCard(slot=2, sck=12, mosi=11, miso=13, cs=10)
        os.mount(sd, "/sd")
        print("boot: SD mounted at /sd")
        return True
    except Exception as e:
        print("boot: SD mount skipped ({})".format(e))
        return False


def _connect_wifi(ssid: str, password: str, timeout_s: float = 30.0) -> bool:
    """Connect to ``ssid``. Returns True once associated. Caller
    should bail out / sleep / blink LED on False — there's no
    point booting the agent stack without network."""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if wlan.isconnected():
        print("boot: WiFi already connected:", wlan.ifconfig())
        return True
    print("boot: WiFi connecting to", ssid)
    wlan.connect(ssid, password)
    deadline = time.time() + timeout_s
    while not wlan.isconnected() and time.time() < deadline:
        time.sleep_ms(250)
    if not wlan.isconnected():
        print("boot: WiFi connect timed out after {}s".format(timeout_s))
        return False
    print("boot: WiFi up:", wlan.ifconfig())
    return True


def _sync_clock() -> None:
    """NTP sync so TLS works. Skipped quietly if the network can't
    reach the NTP server — TLS will fail at ``notBefore`` and
    surface the real problem."""
    try:
        import ntptime

        ntptime.settime()
        print("boot: clock synced via NTP")
    except Exception as e:
        print("boot: NTP sync failed ({})".format(e))


# ── Boot sequence ──────────────────────────────────────────────


_mount_sd()

try:
    import secrets  # type: ignore[import-not-found]
except ImportError:
    print("boot: secrets.py missing — copy secrets.py.example to /secrets.py")
    secrets = None  # type: ignore[assignment]

if secrets is not None:
    if _connect_wifi(secrets.WIFI_SSID, secrets.WIFI_PASSWORD):
        _sync_clock()
