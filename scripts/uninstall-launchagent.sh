#!/bin/bash
# Stop a Foreman service and remove its per-user LaunchAgent (no sudo). Keeps ~/.foreman (DB,
# logs, artifacts, and the phone's secrets: `python3 -m foreman.cli phone unpair` removes those).
#
#   scripts/uninstall-launchagent.sh           # gpud
#   scripts/uninstall-launchagent.sh phone     # the phone service
#
# Runs gpud has in flight are aborted by mlx-serve when its connections close; a later gpud
# records them as interrupted and fires their on_failure continuations. Stopping the phone
# service only stops the messages: nothing it sent or recorded is lost.
set -euo pipefail

SERVICE="gpud"
for arg in "$@"; do
  case "$arg" in
    gpud|phone) SERVICE="$arg" ;;
    *) echo "usage: $0 [gpud|phone]" >&2; exit 2 ;;
  esac
done

LABEL="com.github.ggaabe.foreman.$SERVICE"
DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"

if [ "$(id -u)" -eq 0 ]; then
  echo "refusing to run as root: Foreman's services are per-user LaunchAgents (no sudo)" >&2
  exit 1
fi
if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
  launchctl bootout "$DOMAIN/$LABEL"
  for _ in $(seq 1 60); do  # bootout returns before launchd has removed the service
    launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1 || break
    sleep 0.5
  done
  echo "stopped $LABEL"
else
  echo "$LABEL is not loaded"
fi
if [ -f "$DEST" ]; then
  rm -f "$DEST"
  echo "removed $DEST"
fi
