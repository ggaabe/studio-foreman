#!/bin/bash
# Stop gpud and remove its per-user LaunchAgent (no sudo). Keeps ~/.foreman (DB, logs, artifacts).
# Runs gpud has in flight are aborted by mlx-serve when its connections close; a later gpud
# records them as interrupted and fires their on_failure continuations.
set -euo pipefail

LABEL="com.github.ggaabe.foreman.gpud"
DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"

if [ "$(id -u)" -eq 0 ]; then
  echo "refusing to run as root: gpud is a per-user LaunchAgent (no sudo)" >&2
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
