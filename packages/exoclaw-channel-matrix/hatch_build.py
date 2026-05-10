"""Hatch build hook — materialize codemod outputs before packaging.

Imports `exoclaw_channel_codemod` (declared as a build dep in pyproject.toml
under [tool.hatch.build.hooks.custom].dependencies) and runs it against
this package's vendor/ snapshot. The wheel ends up containing channel.py
even though it's gitignored in the source tree.
"""

from pathlib import Path

from exoclaw_channel_codemod import regenerate
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CodemodBuildHook(BuildHookInterface):
    PLUGIN_NAME = "codemod"

    def initialize(self, version: str, build_data: dict) -> None:
        regenerate(Path(self.root))
