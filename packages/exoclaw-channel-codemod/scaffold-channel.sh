#!/usr/bin/env bash
# Bootstrap a new channel package by snapshotting upstream source + tests
# at the current HEAD of the HKUDS repo. After scaffolding, run
#   bash packages/exoclaw-channel-codemod/sync.sh <name> --apply
# to generate the channel module and tests.
set -euo pipefail

# Repo root: two levels up from this script
# (packages/exoclaw-channel-codemod/scaffold-channel.sh).
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
UPSTREAM=${UPSTREAM:?set UPSTREAM to a HKUDS/nanobot checkout}
NAME=${1:?usage: scaffold-channel.sh <channel-name>}
DEPS=${2:-}  # optional: extra runtime deps for pyproject (comma-separated)

cd "$UPSTREAM"
SHA=$(git rev-parse HEAD)

PKG="${REPO_ROOT}/packages/exoclaw-channel-${NAME}"
mkdir -p "${PKG}/exoclaw_channel_${NAME}" "${PKG}/vendor" "${PKG}/tests"

git show "${SHA}:nanobot/channels/${NAME}.py" > "${PKG}/vendor/upstream.py"
git show "${SHA}:tests/channels/test_${NAME}_channel.py" > "${PKG}/vendor/upstream_test.py" 2>/dev/null || rm -f "${PKG}/vendor/upstream_test.py"
echo "$SHA" > "${PKG}/vendor/SHA"

# Bare __init__.py — actual class names get re-exported once channel.py is generated
cat > "${PKG}/exoclaw_channel_${NAME}/__init__.py" <<EOF
from .channel import *  # noqa: F401,F403
EOF

# Bare pyproject.toml — extras to be filled in per-channel
deps_array=""
if [[ -n "$DEPS" ]]; then
    IFS=',' read -ra arr <<< "$DEPS"
    for d in "${arr[@]}"; do
        deps_array+="    \"${d}\",\n"
    done
fi
cat > "${PKG}/pyproject.toml" <<EOF
[project]
name = "exoclaw-channel-${NAME}"
version = "0.1.0"
description = "${NAME} channel for exoclaw — vendored from HKUDS/nanobot via codemod"
requires-python = ">=3.11"
dependencies = [
    "exoclaw>=0.28.0",
    "exoclaw-nanobot-compat>=0.1.0",
    "loguru>=0.7",
    "pydantic>=2",
$(echo -e "$deps_array")
]
EOF

# conftest so tests find the package
cat > "${PKG}/tests/conftest.py" <<EOF
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
EOF

echo "scaffolded ${PKG}  (SHA=${SHA:0:8})"
