#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <mode> <scp-destination>"
  echo "Modes:"
  echo "  local-to-remote   Sync generate_scripts/, job.sh, and model/ to the remote server"
  echo "  remote-to-local   Sync synthesis/ from the remote server to local"
  echo "Example:"
  echo "  $(basename "$0") local-to-remote user@host:/path/to/synthetic_time_series_benchmarking/"
  echo "  $(basename "$0") remote-to-local user@host:/path/to/synthetic_time_series_benchmarking/"
  exit 1
}

if [[ $# -ne 2 ]]; then
  usage
fi

MODE="$1"
REMOTE="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sync_local_to_remote() {
  local items=(
    "generate_scripts"
    "job.sh"
    "model"
  )

  for item in "${items[@]}"; do
    if [[ ! -e "${SCRIPT_DIR}/${item}" ]]; then
      echo "Error: ${item} not found at ${SCRIPT_DIR}" >&2
      exit 1
    fi
  done

  echo "Syncing local to remote: ${REMOTE}"
  # Use rsync (incremental) and skip nested git metadata and macOS junk.
  # Excluding .git avoids "Permission denied" on read-only git pack objects
  # that already exist on the remote, and keeps the transfer small.
  rsync -av \
    --exclude='.git' \
    --exclude='.DS_Store' \
    "${SCRIPT_DIR}/generate_scripts" \
    "${SCRIPT_DIR}/job.sh" \
    "${SCRIPT_DIR}/model" \
    "${REMOTE}"
}

sync_remote_to_local() {
  local remote_synthesis="${REMOTE%/}"

  echo "Syncing remote to local: ${remote_synthesis} -> ${SCRIPT_DIR}/synthesis/"
  mkdir -p "${SCRIPT_DIR}/synthesis"
  # rsync with a trailing slash on the source copies the *contents* of the
  # remote synthesis/ into the local synthesis/ (no extra nested level, which
  # scp -r would create). --exclude skips macOS junk.
  rsync -av --exclude='.DS_Store' "${remote_synthesis}/" "${SCRIPT_DIR}/synthesis/"
}

case "${MODE}" in
  local-to-remote)
    sync_local_to_remote
    ;;
  remote-to-local)
    sync_remote_to_local
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage
    ;;
esac
