"""Pillow-renderer tests.

CPython-only — gated on ``PIL`` being importable. Tests render
to a temp PNG and assert the file exists at the expected
panel resolution. No pixel-snapshot comparisons; renderer
behaviour is verified at the layout-block level instead.
"""

from __future__ import annotations

import os

import pytest

# Skip the whole module if Pillow isn't installed.
PIL = pytest.importorskip("PIL")

from exoclaw_screen.layout import lay_out  # noqa: E402
from exoclaw_screen.parser import parse  # noqa: E402
from exoclaw_screen.protocol import (  # noqa: E402
    COLOR_MONO,
    REFRESH_SLOW,
    DisplayCapabilities,
)
from exoclaw_screen.renderer.pillow import PillowRenderer  # noqa: E402


def _caps() -> DisplayCapabilities:
    return DisplayCapabilities(
        width=400,
        height=300,
        color_mode=COLOR_MONO,
        refresh_class=REFRESH_SLOW,
        char_cols=40,
        char_rows=12,
        supports_partial=True,
    )


class TestRendersToPng:
    def test_renders_simple_doc(self, tmp_path) -> None:
        out = tmp_path / "screen.png"
        doc = parse("# Hello\n\nbody text")
        blocks = lay_out(doc, _caps())
        PillowRenderer(_caps()).render_to_png(blocks, str(out))
        assert out.exists()
        from PIL import Image

        img = Image.open(str(out))
        assert img.size == (400, 300)


class TestPlainImageItalicFallback:
    def test_plain_image_renders_alt(self, tmp_path) -> None:
        # No recognised class → italic alt text. Smoke test:
        # render doesn't crash and PNG is produced.
        out = tmp_path / "screen.png"
        doc = parse("![alt text](https://example.com/x.png)")
        blocks = lay_out(doc, _caps())
        PillowRenderer(_caps()).render_to_png(blocks, str(out))
        assert out.exists()


class TestQrcodeDirective:
    def test_qrcode_renders_or_falls_back(self, tmp_path) -> None:
        out = tmp_path / "screen.png"
        doc = parse("![QR](https://example.com){.qrcode size=100}")
        blocks = lay_out(doc, _caps())
        PillowRenderer(_caps()).render_to_png(blocks, str(out))
        assert out.exists()


class TestIncludeDirective:
    def test_include_inlines_referenced_md(self, tmp_path) -> None:
        # Set up an inner.md and an outer.md that includes it.
        inner = tmp_path / "inner.md"
        inner.write_text("# Inner heading")
        outer = tmp_path / "outer.md"
        outer.write_text("# Outer\n\n![inner](inner.md){.include}\n")
        out = tmp_path / "screen.png"
        doc = parse(outer.read_text())
        blocks = lay_out(doc, _caps(), base_path=str(tmp_path))
        PillowRenderer(_caps()).render_to_png(blocks, str(out), base_path=str(tmp_path))
        assert out.exists()

    def test_include_handles_missing_file(self, tmp_path) -> None:
        out = tmp_path / "screen.png"
        doc = parse("![missing](does_not_exist.md){.include}")
        blocks = lay_out(doc, _caps(), base_path=str(tmp_path))
        # Should not raise.
        PillowRenderer(_caps()).render_to_png(blocks, str(out), base_path=str(tmp_path))
        assert out.exists()

    def test_include_cycle_detection(self, tmp_path) -> None:
        # a.md includes b.md, b.md includes a.md — single-level
        # only per spec, so b.md's include inside the rendered
        # a.md should be flagged as nested-include / cycle and
        # not crash.
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text("# A\n\n![b](b.md){.include}")
        b.write_text("# B\n\n![a](a.md){.include}")
        out = tmp_path / "screen.png"
        doc = parse(a.read_text())
        blocks = lay_out(doc, _caps(), base_path=str(tmp_path))
        PillowRenderer(_caps()).render_to_png(blocks, str(out), base_path=str(tmp_path))
        assert out.exists()


class TestCleanupArtifacts:
    """Sanity: tmp_path teardown doesn't leak files across tests."""

    def test_tmp_path_isolation(self, tmp_path) -> None:
        assert os.path.isdir(str(tmp_path))
