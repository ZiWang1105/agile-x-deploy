#!/usr/bin/env bash
# Run in the foreground or manage a session-scoped systemd user service.
set -euo pipefail
cd "$(dirname "$0")/.."
project_dir="$PWD"
unit=agilex-zed-stream
action="${1:-run}"
if [[ $# -gt 0 ]]; then shift; fi

case "$action" in
  stop)
    exec systemctl --user stop "$unit"
    ;;
  status)
    exec systemctl --user status "$unit" --no-pager
    ;;
  logs)
    exec journalctl --user -u "$unit" -n 50 --no-pager
    ;;
  run|start|restart) ;;
  *)
    echo "Usage: $0 {run|start|restart|stop|status|logs} [--device PATH] [--port PORT]" >&2
    exit 2
    ;;
esac

if [[ ! -x .venv-camera/bin/python ]]; then
  echo 'Run ./scripts/setup_zed_stream.sh first.' >&2
  exit 1
fi

if [[ "$action" == run ]]; then
  exec .venv-camera/bin/python -u scripts/stream_zed.py "$@"
fi

if systemctl --user is-active --quiet "$unit"; then
  if [[ "$action" == restart ]]; then
    systemctl --user stop "$unit"
  else
    echo 'ZED stream is already running. Use restart to apply new options.'
    exit 0
  fi
fi

systemd-run --user --unit="$unit" --collect \
  --description='ZED 2i local live color stream' \
  --property="WorkingDirectory=$project_dir" \
  "$project_dir/.venv-camera/bin/python" -u "$project_dir/scripts/stream_zed.py" "$@"
echo 'Default preview URL: http://127.0.0.1:8766 (use your chosen port with --port).'
