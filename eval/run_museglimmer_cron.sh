#!/usr/bin/env bash
# Cron wrapper for the sparkinfer Muse Glimmer PR eval bot:
#
#   30 * * * * /home/autotiny/Desktop/sparkinfer/eval/run_museglimmer_cron.sh >> /tmp/sparkinfer_museglimmer_bot.log 2>&1
#   Installed 2026-08-11, offset to :30 (not :00, same as the DFlash bot) so the two don't compete
#   for /tmp/sparkinfer_bot.lock every tick — flock's 120s wait is meant for short overlaps (e.g. a
#   dashboard sync script), not two multi-minute full GPU eval runs racing each other every hour.
#
# Policy (same as the AR + DFlash bots):
#   • Pinned eval box only; never rent / never start from cron when down.
#   • Shares /tmp/sparkinfer_bot.lock with run_dflash_cron.sh (and the AR bot / sparkinfer-web
#     dashboard-sync crons) — CRITICAL: this MUST be the SAME lock file, since every one of these
#     bots drives the ONE pinned GPU on the SAME box and would otherwise race for it if a cron
#     tick overlaps. Wait a bounded amount instead of failing instantly: long enough to outlast a
#     quick sibling-bot tick, short enough to still bail if something is genuinely stuck.
#   • GPU up → full 128-decode speed+accuracy eval; GPU down → --labels-only.
#   • Auto-merge stays OFF unless SPARKINFER_MUSEGLIMMER_AUTOMERGE=1 is explicitly set (NOT
#     exported here, and NOT in .env.eval by default) — unlike the DFlash cron wrapper, this
#     script deliberately does NOT force an auto-merge env var to 1.
export HOME="${HOME:-/home/autotiny}"
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export VAST_NO_AUTO_PROVISION=1

exec 9>/tmp/sparkinfer_bot.lock
flock -w 120 9 || { echo "[$(date -u +%FT%TZ)] lock held 120s+ — previous bot run still active, skipping museglimmer tick"; exit 0; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1

if [ -f "$REPO_DIR/.env.eval" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$REPO_DIR/.env.eval"
  set +a
fi
export VAST_NO_AUTO_PROVISION=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

git pull -q origin main 2>/dev/null || true

PIN_FILE="${VAST_PIN_FILE:-$HOME/.sparkinfer_pinned_instance}"
INSTANCE_FILE="${VAST_INSTANCE_FILE:-$HOME/.sparkinfer_vast_instance}"
resolve_pin() {
  local v=""
  if [ -f "$PIN_FILE" ]; then
    v="$(tr -d '[:space:]' <"$PIN_FILE" 2>/dev/null || true)"
  fi
  if [ -z "$v" ] || [ "$v" = "0" ]; then
    v="${VAST_DEFAULT_INSTANCE:-${VAST_INSTANCE:-}}"
  fi
  printf '%s' "$v"
}
PINNED_ID="$(resolve_pin)"
if [ "${EVAL_TRANSPORT:-vast}" != "ssh" ] && [ -n "$PINNED_ID" ] && [ "$PINNED_ID" != "0" ]; then
  export VAST_INSTANCE="$PINNED_ID"
  export VAST_DEFAULT_INSTANCE="$PINNED_ID"
  printf '%s\n' "$PINNED_ID" >"$PIN_FILE"
  printf '%s\n' "$PINNED_ID" >"$INSTANCE_FILE"
fi

BOT_ARGS=(--repo "${REPO:-gittensor-ai-lab/sparkinfer}")
if [ "${EVAL_TRANSPORT:-vast}" != "ssh" ]; then
  BOT_ARGS+=(--instance "${VAST_INSTANCE:-0}")
fi

gpu_ready() {
  local key="${SSH_KEY:-$HOME/.ssh/speedy}"
  if [ "${EVAL_TRANSPORT:-vast}" = "ssh" ]; then
    local host="${EVAL_SSH_HOST:-}" port="${EVAL_SSH_PORT:-22}" user="${EVAL_SSH_USER:-root}"
    [ -n "$host" ] || return 1
    # IdentitiesOnly=yes: same rationale as run_dflash_cron.sh's gpu_ready — cron has no
    # ssh-agent, so without it a box whose authorized_keys only has SSH_KEY's public half (not
    # some agent identity) fails outright with "Permission denied".
    local err rc
    err="$(ssh -i "$key" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 \
        -o StrictHostKeyChecking=accept-new -p "$port" "$user@$host" 'true' 2>&1)"
    rc=$?
    [ "$rc" -eq 0 ] || echo "gpu_ready: ssh to $user@$host:$port failed (exit=$rc): $err" >&2
    return "$rc"
  fi
  local iid="${VAST_INSTANCE:-}"
  [ -n "$iid" ] && [ "$iid" != "0" ] || return 1
  command -v vastai >/dev/null 2>&1 || return 1
  local raw st ip port
  raw="$(vastai show instance "$iid" --raw 2>/dev/null)" || return 1
  read -r st ip port < <(python3 -c "
import json, sys
d = json.loads(sys.stdin.read() or '{}')
st = d.get('actual_status') or ''
ip = d.get('public_ipaddr') or ''
ports = d.get('ports') or {}
p = ((ports.get('22/tcp') or [{}])[0] or {}).get('HostPort') or ''
print(st, ip, p)
" <<<"$raw")
  [ "$st" = "running" ] && [ -n "$ip" ] && [ -n "$port" ] || return 1
  ssh -i "$key" -o BatchMode=yes -o ConnectTimeout=10 \
      -o StrictHostKeyChecking=accept-new -p "$port" "root@$ip" 'true' 2>/dev/null
}

GPU_LABEL="${EVAL_SSH_HOST:-ssh}"
[ "${EVAL_TRANSPORT:-vast}" = "ssh" ] || GPU_LABEL="${VAST_INSTANCE:-?}"

TS="$(date -u +%FT%TZ)"
if gpu_ready; then
  echo "[$TS] sparkinfer Muse Glimmer bot — pinned GPU $GPU_LABEL up — full eval (AUTOMERGE=${SPARKINFER_MUSEGLIMMER_AUTOMERGE:-0})"
  python3 eval/pr_museglimmer_bot.py "${BOT_ARGS[@]}"
else
  echo "[$TS] sparkinfer Muse Glimmer bot — pinned GPU $GPU_LABEL down — labels only"
  python3 eval/pr_museglimmer_bot.py "${BOT_ARGS[@]}" --labels-only
fi
