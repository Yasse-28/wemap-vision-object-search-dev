#!/usr/bin/env bash
# Coding/review handoff lock for two agents sharing this working tree — e.g.
# Codex as coder, Claude as reviewer. Advisory only: it helps exactly as much
# as every agent checks it before touching files, nothing in git or the
# filesystem enforces it.
#
# State machine, one lock at a time:
#
#   (unlocked) --acquire--> coding --ready--> review_ready --review-start--> reviewing
#       ^                                                          |   |
#       |                                                       reject approve
#       |                                                          |   |
#       +----------------------- coding <------------------------+   |
#       +------------------------------ (unlocked) <-------------------+
#
# `release --force` breaks the lock from any state (abort/recovery path).
#
# Why `.agents/lock.d` and not a lock *file*: `mkdir` is atomic even over
# NFS/network mounts, unlike a plain "check then write" on a file. The lock's
# existence is `lock.d` existing; `repo-lock.json` inside it is just metadata,
# safe to rewrite because only the label that holds the lock is allowed to.
#
# Requires `jq` (already on this machine) — used for every read/write so the
# JSON these files carry (parsed by whichever agent's own tooling) is never
# hand-escaped and never corrupted by a partial write.
#
# Usage:
#   scripts/agent-lock.sh acquire <label> "<task>" ["<note>"]
#   scripts/agent-lock.sh heartbeat <label>
#   scripts/agent-lock.sh ready <label> <commit> ["<file1,file2,...>"]
#   scripts/agent-lock.sh review-start <reviewer-label>
#   scripts/agent-lock.sh approve <reviewer-label>
#   scripts/agent-lock.sh reject <reviewer-label> ["<note>"]
#   scripts/agent-lock.sh release <label> [--force]
#   scripts/agent-lock.sh status
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENTS_DIR="$REPO_ROOT/.agents"
LOCK_DIR="$AGENTS_DIR/lock.d"
LOCK_FILE="$LOCK_DIR/repo-lock.json"
READY_FILE="$AGENTS_DIR/review-ready.json"
STALE_SECONDS="${AGENT_LOCK_STALE_SECONDS:-1800}" # 30 min

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required (brew/apt install jq)" >&2
  exit 2
fi

usage() {
  cat >&2 <<EOF
usage: $(basename "$0") acquire <label> "<task>" ["<note>"]
       $(basename "$0") heartbeat <label>
       $(basename "$0") ready <label> <commit> ["<file1,file2,...>"]
       $(basename "$0") review-start <reviewer-label>
       $(basename "$0") approve <reviewer-label>
       $(basename "$0") reject <reviewer-label> ["<note>"]
       $(basename "$0") release <label> [--force]
       $(basename "$0") status
EOF
  exit 2
}

now_iso() { date -u +%Y-%m-%dT%H:%M:%SZ; }

age_seconds() {
  local then_epoch now_epoch
  then_epoch="$(date -u -d "$1" +%s 2>/dev/null || date -u -jf %Y-%m-%dT%H:%M:%SZ "$1" +%s)"
  now_epoch="$(date -u +%s)"
  echo $(( now_epoch - then_epoch ))
}

# Atomic write: same directory, then rename — a reader never sees a partial file.
write_json_atomic() {
  local path="$1" content="$2"
  local tmp
  tmp="$(mktemp "${path}.XXXXXX")"
  printf '%s\n' "$content" > "$tmp"
  mv "$tmp" "$path"
}

require_lock() {
  if [[ ! -d "$LOCK_DIR" || ! -f "$LOCK_FILE" ]]; then
    echo "error: repo is not locked" >&2
    exit 1
  fi
}

print_status() {
  if [[ ! -d "$LOCK_DIR" || ! -f "$LOCK_FILE" ]]; then
    echo "unlocked"
    return 0
  fi
  local state coder reviewer task hb age
  state="$(jq -r '.state' "$LOCK_FILE")"
  coder="$(jq -r '.coder' "$LOCK_FILE")"
  reviewer="$(jq -r '.reviewer' "$LOCK_FILE")"
  task="$(jq -r '.task' "$LOCK_FILE")"
  hb="$(jq -r '.last_heartbeat' "$LOCK_FILE")"
  age="$(age_seconds "$hb")"
  echo "state=$state coder=$coder reviewer=$reviewer task=\"$task\" last_heartbeat=${age}s ago"
  if [[ -f "$READY_FILE" ]]; then
    echo "review-ready: $(jq -c . "$READY_FILE")"
  fi
  if (( age > STALE_SECONDS )); then
    echo "warning: heartbeat is stale (> ${STALE_SECONDS}s) — holder may be gone." >&2
    echo "         recover with: $(basename "$0") release <label-above> --force" >&2
  fi
}

cmd="${1:-}"
case "$cmd" in

  acquire)
    label="${2:-}"; task="${3:-}"; note="${4:-}"
    [[ -z "$label" || -z "$task" ]] && usage
    mkdir -p "$AGENTS_DIR"
    if ! mkdir "$LOCK_DIR" 2>/dev/null; then
      echo "error: repo is locked" >&2
      print_status
      exit 1
    fi
    write_json_atomic "$LOCK_FILE" "$(jq -n \
      --arg state coding --arg coder "$label" --arg task "$task" --arg note "$note" \
      --arg now "$(now_iso)" \
      '{state: $state, coder: $coder, reviewer: null, task: $task, note: $note,
        acquired_at: $now, last_heartbeat: $now}')"
    echo "lock acquired by '$label' (state=coding)"
    ;;

  heartbeat)
    label="${2:-}"; [[ -z "$label" ]] && usage
    require_lock
    holder="$(jq -r 'if .state == "reviewing" then .reviewer else .coder end' "$LOCK_FILE")"
    if [[ "$holder" != "$label" ]]; then
      echo "error: '$label' does not currently hold the lock (holder: '$holder')" >&2
      exit 1
    fi
    write_json_atomic "$LOCK_FILE" "$(jq --arg now "$(now_iso)" '.last_heartbeat = $now' "$LOCK_FILE")"
    echo "heartbeat refreshed"
    ;;

  ready)
    label="${2:-}"; commit="${3:-}"; files="${4:-}"
    [[ -z "$label" || -z "$commit" ]] && usage
    require_lock
    state="$(jq -r '.state' "$LOCK_FILE")"
    coder="$(jq -r '.coder' "$LOCK_FILE")"
    if [[ "$state" != "coding" || "$coder" != "$label" ]]; then
      echo "error: '$label' cannot signal ready from state=$state (coder=$coder)" >&2
      exit 1
    fi
    task="$(jq -r '.task' "$LOCK_FILE")"
    write_json_atomic "$READY_FILE" "$(jq -n \
      --arg commit "$commit" --arg files "$files" --arg task "$task" \
      --arg coder "$label" --arg now "$(now_iso)" \
      '{commit: $commit,
        files: ($files | select(length > 0) | split(",")) // [],
        task: $task, coder: $coder, ready_at: $now}')"
    write_json_atomic "$LOCK_FILE" "$(jq --arg now "$(now_iso)" \
      '.state = "review_ready" | .last_heartbeat = $now' "$LOCK_FILE")"
    echo "marked ready for review (commit=$commit)"
    ;;

  review-start)
    label="${2:-}"; [[ -z "$label" ]] && usage
    require_lock
    state="$(jq -r '.state' "$LOCK_FILE")"
    if [[ "$state" != "review_ready" ]]; then
      echo "error: nothing to review (state=$state)" >&2
      exit 1
    fi
    write_json_atomic "$LOCK_FILE" "$(jq --arg now "$(now_iso)" --arg reviewer "$label" \
      '.state = "reviewing" | .reviewer = $reviewer | .last_heartbeat = $now' "$LOCK_FILE")"
    echo "review started by '$label'"
    [[ -f "$READY_FILE" ]] && echo "review-ready: $(jq -c . "$READY_FILE")"
    ;;

  approve)
    label="${2:-}"; [[ -z "$label" ]] && usage
    require_lock
    state="$(jq -r '.state' "$LOCK_FILE")"
    reviewer="$(jq -r '.reviewer' "$LOCK_FILE")"
    if [[ "$state" != "reviewing" || "$reviewer" != "$label" ]]; then
      echo "error: '$label' cannot approve from state=$state (reviewer=$reviewer)" >&2
      exit 1
    fi
    rm -rf "$LOCK_DIR" "$READY_FILE"
    echo "approved — lock released"
    ;;

  reject)
    label="${2:-}"; note="${3:-}"
    [[ -z "$label" ]] && usage
    require_lock
    state="$(jq -r '.state' "$LOCK_FILE")"
    reviewer="$(jq -r '.reviewer' "$LOCK_FILE")"
    if [[ "$state" != "reviewing" || "$reviewer" != "$label" ]]; then
      echo "error: '$label' cannot reject from state=$state (reviewer=$reviewer)" >&2
      exit 1
    fi
    write_json_atomic "$LOCK_FILE" "$(jq --arg now "$(now_iso)" --arg note "$note" \
      '.state = "coding" | .reviewer = null | .note = $note | .last_heartbeat = $now' "$LOCK_FILE")"
    rm -f "$READY_FILE"
    echo "sent back to coding"
    ;;

  release)
    label="${2:-}"; force="${3:-}"
    [[ -z "$label" ]] && usage
    if [[ ! -d "$LOCK_DIR" ]]; then
      echo "already unlocked"
      exit 0
    fi
    holder="$(jq -r 'if .state == "reviewing" then .reviewer else .coder end' "$LOCK_FILE" 2>/dev/null || echo "?")"
    if [[ "$holder" != "$label" && "$force" != "--force" ]]; then
      echo "error: lock is held by '$holder', not '$label'" >&2
      echo "pass --force to break it anyway" >&2
      exit 1
    fi
    rm -rf "$LOCK_DIR" "$READY_FILE"
    echo "lock released (was held by '$holder')"
    ;;

  status)
    print_status
    ;;

  *)
    usage
    ;;
esac
