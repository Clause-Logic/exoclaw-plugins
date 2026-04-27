"""MicroPython import smoke test for ``exoclaw-provider-openai``.

Pure-Python — no pytest. Driven by the workspace's
``mise run test-micro`` task on a coverage-variant MicroPython
binary.
"""


def test_top_level_imports():
    from exoclaw_provider_openai import Deployment, OpenAIStreamingProvider

    assert callable(OpenAIStreamingProvider)
    # ``Deployment`` is the dual @dataclass / plain class — both
    # branches must accept the documented kwargs.
    d = Deployment(base_url="https://x.test/v1", api_key="k")
    assert d.base_url == "https://x.test/v1"
    assert d.api_key == "k"
    assert d.extra_headers == {}
    assert d.extra_body == {}


def test_provider_module_imports():
    from exoclaw_provider_openai import provider

    # ``_short_tool_id`` uses ``os.urandom`` on MP (no ``secrets``
    # module) — exercise it once to confirm the runtime branch.
    out = provider._short_tool_id()
    assert isinstance(out, str)
    assert len(out) == 9
    # MP strings don't have ``.isalnum()``; check membership in
    # the alphabet directly.
    for c in out:
        assert c in provider._ALNUM


def test_streaming_provider_constructs():
    from exoclaw_provider_openai import Deployment, OpenAIStreamingProvider

    p = OpenAIStreamingProvider(
        default_model="m1",
        deployments={"m1": Deployment(base_url="https://x.test/v1", api_key="k")},
    )
    assert p.get_default_model() == "m1"
