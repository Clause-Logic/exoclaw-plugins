"""Fixture-driven tests for ``exoclaw_tools_web.convert``.

Each ``tests/fixtures/<name>.html`` / ``<name>.md`` pair is loaded,
the converter run against the HTML, and the result compared to the
expected Markdown.

A handful of fixtures were extracted from turndown cases that use
non-default options (``linkStyle: 'referenced'``,
``codeBlockStyle: 'fenced'``, ``headingStyle: 'atx'``, custom bullet
markers, ``preformattedCode: true``, ``hr``/``br`` overrides). Our
converter ships with turndown's defaults baked in — those fixtures
are listed in :data:`KNOWN_FAILURES` and asserted to NOT match (so
if we ever start matching one, the test surfaces it).

Plus a small set of edge-case mismatches we don't currently handle
(turndown's exact whitespace edge in code blocks with embedded
fence sequences, etc.) — those join ``KNOWN_FAILURES`` too.
"""

from __future__ import annotations

import os

from exoclaw_tools_web import convert

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


# ── Known-fail registry ─────────────────────────────────────────
#
# These fixtures all derive from turndown cases that override at
# least one option from the default. Listing them by name keeps the
# pass-rate metric meaningful — every other fixture in the dir is
# expected to round-trip cleanly.

OPTION_DRIVEN = {
    # ``headingStyle: 'atx'`` (default is ``setext``).
    "h1_as_atx",
    "h2_as_atx",
    # ``hr: '- - -'`` / ``br: '\\'`` overrides.
    # (No corresponding fixture name in the corpus that we ship
    # under those data-names, but listed for completeness.)
    # ``codeBlockStyle: 'fenced'``.
    "fenced_pre_code_block",
    "fenced_pre_code_block_with_language",
    "pre_code_block_fenced_with",
    "triple_ticks_inside_code",
    "triple_tildes_inside_code",
    "four_ticks_inside_code",
    "empty_line_in_start_end_of_code_block",
    # ``linkStyle: 'referenced'`` (+ collapsed/shortcut variants).
    "a_reference",
    "a_reference_with_title",
    "a_reference_with_space_in_url",
    "a_reference_with_collapsed_style",
    "a_reference_with_shortcut_style",
    # ``bulletListMarker: '-'``.
    "ul_with_custom_bullet",
    # ``preformattedCode: true``.
    # ``loosely_surrounded`` / ``tightly_surrounded`` actually
    # round-trip cleanly under the default options too — turndown's
    # ``preformattedCode`` option only matters for the cases below
    # where the inner whitespace would otherwise be flattened.
    "preformatted_code_with_leading_whitespace",
    "preformatted_code_with_trailing_whitespace",
    "preformatted_code_with_newlines",
}


# Edge cases we don't try to perfectly replicate. Documented so a
# regression toward correctness never goes unnoticed (we'd see the
# fixture flip from "expected fail" to "passing" and could move it
# out).
EDGE_CASE_FAILURES: set[str] = set()
# Reserved for case-by-case edge fails discovered during dev.

KNOWN_FAILURES = OPTION_DRIVEN | EDGE_CASE_FAILURES


def _fixture_names() -> list[str]:
    names = set()
    for fn in os.listdir(FIXTURES):
        if fn.endswith(".html"):
            names.add(fn[:-5])
    return sorted(names)


def _load(name: str) -> tuple[str, str]:
    with open(os.path.join(FIXTURES, name + ".html"), encoding="utf-8") as f:
        html = f.read()
    with open(os.path.join(FIXTURES, name + ".md"), encoding="utf-8") as f:
        expected = f.read()
    return html, expected


def test_fixtures_summary() -> None:
    """Walk every fixture, run the converter, count pass/fail.

    Fails the test if the pass rate dips below 60% of total fixtures
    (the bar set when this package was scoped). Prints a per-fixture
    summary on failure to make diffs greppable.
    """
    names = _fixture_names()
    total = len(names)
    passes: list[str] = []
    failures: list[tuple[str, str, str]] = []

    for name in names:
        html, expected = _load(name)
        # Strip trailing newline from expected — fixture files
        # commonly end with one final ``\n`` that's a side effect of
        # how the corpus was emitted, not part of the expected
        # output.
        if expected.endswith("\n") and not expected.endswith("\n\n"):
            expected = expected[:-1]
        try:
            got = convert(html)
        except Exception as e:  # noqa: BLE001
            failures.append((name, "<EXCEPTION: " + repr(e) + ">", expected))
            continue
        if got == expected:
            passes.append(name)
        else:
            failures.append((name, got, expected))

    # Compute the "real" pass count: a known-failure that passes
    # gets credited; a known-failure that fails does NOT count
    # against us.
    real_failures = [f for f in failures if f[0] not in KNOWN_FAILURES]
    real_total = total - len([n for n in KNOWN_FAILURES if n in names])

    pass_rate = (len(passes) / total) if total else 0.0
    expected_rate = (len(passes) / real_total) if real_total else 0.0

    print("\n=== exoclaw-tools-web fixture summary ===")
    print(
        "total={} pass={} fail={} known_fail={} pass_rate={:.1%} of_default_options={:.1%}".format(
            total,
            len(passes),
            len(failures),
            len([n for n in KNOWN_FAILURES if n in names]),
            pass_rate,
            expected_rate,
        )
    )
    if real_failures:
        print("\nUnexpected failures (first 30):")
        for name, got, expected in real_failures[:30]:
            print("--- " + name + " ---")
            print("EXPECTED: " + repr(expected))
            print("GOT     : " + repr(got))

    # Bar: 60% overall pass rate.
    assert pass_rate >= 0.60, "pass rate {:.1%} below 60% bar — got {}/{}".format(
        pass_rate, len(passes), total
    )
    # Plus: every non-known-failure must pass.
    assert not real_failures, "{} unexpected failures (see printed summary)".format(
        len(real_failures)
    )
