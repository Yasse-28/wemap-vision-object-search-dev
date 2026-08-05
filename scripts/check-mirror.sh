#!/usr/bin/env bash
# Wrapper: the mirror gate lives with the code it guards, in the submodule.
#
# Kept here so the documented gate list still runs from the repo root, and so
# WEMAP_VISION_BACKEND from this repo's .env reaches it. Everything else — the path
# mapping, the exclusions, the backend discovery — is the submodule's business.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

exec "$REPO_ROOT/third_party/object_search/scripts/check-mirror.sh" "$@"
