#!/usr/bin/env bash
# Per-channel package layout:
#   packages/exoclaw-channel-<name>/
#     pyproject.toml
#     exoclaw_channel_<name>/{__init__.py, channel.py}   ← channel.py GENERATED
#     vendor/{upstream.py, upstream_test.py, SHA}        ← raw HKUDS snapshots
#     patches/                                            ← optional, see below
#       00NN-source-<slug>.patch  applied to channel.py after codemod
#       00NN-test-<slug>.patch    applied to test_channel.py after codemod
#     tests/test_channel.py                              ← GENERATED
#
# Pipeline: codemod(vendor/upstream.py) → patches/*-source-* → channel.py
#           codemod(vendor/upstream_test.py) → patches/*-test-* → tests/
#
# Patches are optional — most channels need none. Use one when the codemod
# can't generalize a channel-specific tweak. Generate via:
#   git diff > packages/<pkg>/patches/00NN-<source|test>-<slug>.patch
#
# Upgrade workflow:
#   1. Bump packages/exoclaw-channel-<name>/vendor/SHA to a new HKUDS commit
#   2. UPSTREAM=~/hkuds-nanobot bash packages/exoclaw-channel-codemod/sync.sh <name> --apply
#   3. If a patch fails, regenerate it against the new codemod output
#   4. Run pytest, commit
set -euo pipefail

# Repo root: two levels up from this script
# (packages/exoclaw-channel-codemod/sync.sh).
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
UPSTREAM=${UPSTREAM:?set UPSTREAM to a HKUDS/nanobot checkout, e.g. UPSTREAM=~/hkuds-nanobot}
NAME=${1:?usage: sync.sh <channel-name> [--apply]}
APPLY=${2:-}

PKG_DIR="${REPO_ROOT}/packages/exoclaw-channel-${NAME}"
VENDOR="${PKG_DIR}/vendor"
PATCHES="${PKG_DIR}/patches"
SHA=$(cat "${VENDOR}/SHA")

cd "$UPSTREAM"
new_src=$(mktemp); new_test=$(mktemp)
git show "${SHA}:nanobot/channels/${NAME}.py" > "$new_src"
git show "${SHA}:tests/channels/test_${NAME}_channel.py" > "$new_test" 2>/dev/null \
    || { echo "  (no upstream test for ${NAME})" >&2; new_test=""; }

src_diff=$(diff -q "$new_src" "${VENDOR}/upstream.py" >/dev/null 2>&1 && echo same || echo differ)
if [[ -n "$new_test" && -f "${VENDOR}/upstream_test.py" ]]; then
    test_diff=$(diff -q "$new_test" "${VENDOR}/upstream_test.py" >/dev/null 2>&1 && echo same || echo differ)
else
    test_diff="missing"
fi
echo "[${NAME}] @ ${SHA:0:8}  source=${src_diff}  test=${test_diff}"

if [[ "$APPLY" != "--apply" ]]; then
    rm -f "$new_src" "$new_test"
    exit 0
fi

cp "$new_src" "${VENDOR}/upstream.py"
[[ -n "$new_test" ]] && cp "$new_test" "${VENDOR}/upstream_test.py"
cd "$REPO_ROOT"

CHAN_OUT="${PKG_DIR}/exoclaw_channel_${NAME}/channel.py"
TEST_OUT="${PKG_DIR}/tests/test_channel.py"
WARN=$(mktemp)

# 1. Codemod source
python3 "$SCRIPT_DIR/codemod.py" source "${VENDOR}/upstream.py" > "$CHAN_OUT" 2>>"$WARN"

# 2. Apply source patches in order
src_patch_count=0
if [[ -d "$PATCHES" ]]; then
    for p in "$PATCHES"/*-source-*.patch; do
        [[ -e "$p" ]] || continue
        src_patch_count=$((src_patch_count + 1))
        if ! ( cd "$(dirname "$CHAN_OUT")" && patch --silent --no-backup-if-mismatch -p1 < "$p" ); then
            echo "  ✗ source patch FAILED: $(basename "$p")"
            echo "    upstream code likely moved; regenerate this patch against the new codemod output"
            exit 1
        fi
    done
fi

# 3. Codemod test (if upstream has one)
test_patch_count=0
if [[ -n "$new_test" ]]; then
    python3 "$SCRIPT_DIR/codemod.py" test "${VENDOR}/upstream_test.py" > "$TEST_OUT" 2>>"$WARN"
    if [[ -d "$PATCHES" ]]; then
        for p in "$PATCHES"/*-test-*.patch; do
            [[ -e "$p" ]] || continue
            test_patch_count=$((test_patch_count + 1))
            if ! ( cd "$(dirname "$TEST_OUT")" && patch --silent --no-backup-if-mismatch -p1 < "$p" ); then
                echo "  ✗ test patch FAILED: $(basename "$p")"
                echo "    upstream code likely moved; regenerate this patch against the new codemod output"
                exit 1
            fi
        done
    fi
fi

if [[ -s "$WARN" ]]; then
    echo "  WARNINGS:"
    sed 's/^/    /' "$WARN"
fi
rm -f "$WARN"

patches_msg=""
[[ $src_patch_count -gt 0 || $test_patch_count -gt 0 ]] && patches_msg=" (+${src_patch_count} src patch, +${test_patch_count} test patch)"
echo "  ✓ regenerated${patches_msg}"
rm -f "$new_src" "$new_test"
