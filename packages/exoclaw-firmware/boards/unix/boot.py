"""Unix-port boot script — runs once before ``main.py``.

The MicroPython unix port already has working network and
filesystem (the host's), so there's nothing equivalent to
``machine.SDCard`` mount or ``network.WLAN.connect`` to do here.
This file exists so the staging task has the same
``boards/<board>/boot.py`` + ``main.py`` shape as the chip
variants.

Used to run the agent locally for development / CI smoke tests
without flashing real hardware. Drive it with
``mise run sim`` from the package directory.
"""

print("boot: unix-port — host network + filesystem in use")
