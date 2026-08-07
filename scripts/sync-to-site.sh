#!/usr/bin/env bash
# Copy the deployable app into the j4den.com site repo, which serves it at
# j4den.com/nnvisual/. Run from anywhere; commit the site repo afterwards.
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
site="${J4DEN_REPO:-$here/../j4den}"
dest="$site/frontend/public/nnvisual"

if [ ! -d "$site/frontend/public" ]; then
    echo "site repo not found at $site (set J4DEN_REPO to override)" >&2
    exit 1
fi

mkdir -p "$dest"
rsync -a --delete --exclude .DS_Store "$here/public/" "$dest/"
echo "synced public/ -> $dest"
