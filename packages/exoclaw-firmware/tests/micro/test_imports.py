"""MicroPython import smoke test for ``exoclaw-firmware``.

Pure-Python — no pytest. Driven by the workspace's
``mise run test-micro`` task on a coverage-variant MicroPython
binary.

The firmware package's job on the chip is to wire core +
conversation + provider into a turn-driving pair. The MP smoke
test verifies the import graph clears cleanly; the real network
path runs on hardware via ``mise run flash`` (see
``packages/exoclaw-firmware/mise.toml``).
"""


def test_top_level_imports():
    from exoclaw_firmware import SerialChannel, build_agent, run_demo, run_serial_app

    assert callable(build_agent)
    assert callable(run_demo)
    assert callable(run_serial_app)
    assert callable(SerialChannel)


def test_app_module_imports():
    from exoclaw_firmware import app

    # ``build_agent`` and ``run_demo`` are re-exported via
    # ``__init__`` but the underlying module must also import
    # cleanly.
    assert callable(app.build_agent)
    assert callable(app.run_demo)


def test_serial_channel_protocol_shape():
    from exoclaw_firmware import SerialChannel

    ch = SerialChannel()
    assert ch.name == "serial"
    # Channel protocol surface — start/stop/send must all exist.
    for method in ("start", "stop", "send"):
        assert callable(getattr(ch, method))
