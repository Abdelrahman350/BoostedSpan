#!/usr/bin/env bash
# Clone the organizers' Daleel2026 data repo into data/raw/ (gitignored).
# Idempotent by default: skips the clone if data/raw/Daleel2026 already exists.
# Pass --force to remove and re-clone.
set -euo pipefail

REPO_URL="https://github.com/Argmining/Daleel2026.git"
DEST="dataset/raw/Daleel2026"

if [[ "${1:-}" == "--force" ]] && [[ -d "$DEST" ]]; then
    echo "Removing existing $DEST (--force)"
    rm -rf "$DEST"
fi

if [[ -d "$DEST" ]]; then
    echo "$DEST already exists, skipping clone (use --force to re-clone)"
else
    mkdir -p "$(dirname "$DEST")"
    git clone --depth 1 "$REPO_URL" "$DEST"
fi
