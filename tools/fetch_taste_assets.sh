#!/usr/bin/env bash
# The taste-skill asset library (~14 MB) is not committed; its checker scripts need it. Fetch it at the vendored commit.
set -eu
T=$(mktemp -d); git clone -q --filter=blob:none https://github.com/Leonxlnx/taste-skill "$T" && git -C "$T" checkout -q f6b7657
D="$(cd "$(dirname "$0")/.." && pwd)/.claude/skills/taste-skill"
cp -r "$T"/skills/taste-skill/assets/*/ "$D/assets/" && cp -r "$T/skills/taste-skill/library" "$D/" && rm -rf "$T"
echo "taste-skill assets restored in $D"
