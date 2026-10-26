#!/bin/bash
# Install gpud as a per-user LaunchAgent (no sudo) and start it.
#
#   scripts/install-launchagent.sh             # render, validate, bootstrap into gui/<uid>
#   scripts/install-launchagent.sh --dry-run   # only print the rendered plist
#
# FOREMAN_PYTHON overrides the interpreter (default /usr/bin/python3, 3.9 or newer).
set -euo pipefail

LABEL="com.github.ggaabe.foreman.gpud"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${FOREMAN_PYTHON:-/usr/bin/python3}"
TEMPLATE="$REPO/deploy/$LABEL.plist"

if [ "$(id -u)" -eq 0 ]; then
  echo "refusing to run as root: gpud is a per-user LaunchAgent (no sudo)" >&2
  exit 1
fi
case "$REPO$HOME$PY" in
  *"#"*) echo "paths containing '#' are not supported" >&2; exit 1 ;;
esac

render() {
  sed -e "s#__PYTHON__#$PY#g" -e "s#__REPO__#$REPO#g" -e "s#__HOME__#$HOME#g" "$TEMPLATE"
}

if [ "${1:-}" = "--dry-run" ]; then
  render
  exit 0
fi

"$PY" -c 'import sys; assert sys.version_info >= (3, 9), sys.version' || {
  echo "$PY is older than Python 3.9" >&2; exit 1; }

mkdir -p "$HOME/.foreman/logs" "$HOME/Library/LaunchAgents"
DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
render > "$DEST.tmp"
plutil -lint "$DEST.tmp" >/dev/null
mv "$DEST.tmp" "$DEST"

DOMAIN="gui/$(id -u)"
# Replace a running instance (SIGTERM; gpud finishes its tick and exits well inside ExitTimeOut).
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$DEST"
echo "installed $DEST"
launchctl print "$DOMAIN/$LABEL" | sed -n '1,25p'
echo "logs: $HOME/.foreman/logs/gpud.log (JSON lines); state: python3 -m foreman.cli status"
