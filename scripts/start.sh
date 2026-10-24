#!/usr/bin/env bash
# Start ComfyUI, the Comfy API proxy and the LiDAR Lab gallery app.
#
#   COMFYUI_DIR=~/ComfyUI ./scripts/start.sh
#
# Optional: COMFY_PYTHON (python for ComfyUI), APP_PYTHON (python with app/requirements.txt),
#           COMFY_PORT (8188), PROXY_PORT (8189), APP_PORT (7860), COMFY_ARGS (extra ComfyUI flags)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
COMFYUI_DIR="${COMFYUI_DIR:?set COMFYUI_DIR to your ComfyUI checkout}"
COMFY_PYTHON="${COMFY_PYTHON:-$( [ -x "$COMFYUI_DIR/.venv/bin/python" ] && echo "$COMFYUI_DIR/.venv/bin/python" || echo python3 )}"
APP_PYTHON="${APP_PYTHON:-python3}"
COMFY_PORT="${COMFY_PORT:-8188}"
PROXY_PORT="${PROXY_PORT:-8189}"
APP_PORT="${APP_PORT:-7860}"
LOGS="$REPO/.logs"
mkdir -p "$LOGS"

wait_for() {  # url name
  for _ in $(seq 1 120); do curl -s -m 2 "$1" >/dev/null 2>&1 && { echo "  $2 is up"; return 0; }; sleep 1; done
  echo "  $2 did not come up, see $LOGS"; exit 1
}

# make sure the node pack is installed
if [ ! -e "$COMFYUI_DIR/custom_nodes/lidar-lab" ]; then
  ln -s "$REPO" "$COMFYUI_DIR/custom_nodes/lidar-lab"
fi

ACCEL=$("$COMFY_PYTHON" -c "import torch; print('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))")
EXTRA="${COMFY_ARGS:-}"
[ "$ACCEL" = "cpu" ] && EXTRA="--cpu $EXTRA"
echo "ComfyUI on $ACCEL (port $COMFY_PORT)"
(cd "$COMFYUI_DIR" && nohup "$COMFY_PYTHON" main.py --listen 127.0.0.1 --port "$COMFY_PORT" --disable-auto-launch $EXTRA \
  > "$LOGS/comfyui.log" 2>&1 & echo $! > "$LOGS/comfyui.pid")
wait_for "http://127.0.0.1:$COMFY_PORT/system_stats" ComfyUI

echo "Comfy API proxy (port $PROXY_PORT)"
nohup "$APP_PYTHON" -m comfy_api_proxy.cli run --comfyui "http://127.0.0.1:$COMFY_PORT" --port "$PROXY_PORT" \
  --comfyui-base-dir "$COMFYUI_DIR" --max-upload-mb 500 > "$LOGS/proxy.log" 2>&1 & echo $! > "$LOGS/proxy.pid"
sleep 2

echo "LiDAR Lab app (port $APP_PORT)"
(cd "$REPO" && COMFY_BASE_URL="http://127.0.0.1:$PROXY_PORT" nohup "$APP_PYTHON" -m uvicorn app.server:app \
  --host 127.0.0.1 --port "$APP_PORT" > "$LOGS/app.log" 2>&1 & echo $! > "$LOGS/app.pid")
wait_for "http://127.0.0.1:$APP_PORT/api/presets" "LiDAR Lab app"
echo
echo "Open http://127.0.0.1:$APP_PORT  (ComfyUI: http://127.0.0.1:$COMFY_PORT)"
echo "Stop everything with: kill \$(cat $LOGS/*.pid)"
