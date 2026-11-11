#!/bin/zsh
# Double-click to open the weather module lab in Chrome.
# Starts lab_server.py, which serves this folder (Web Serial only works on
# http://localhost, not on a file opened from Finder) and lets the page
# autosave its data into runs/. Close this Terminal window, or press Ctrl-C,
# to stop it.
cd "$(dirname "$0")"
PORT=8077
URL="http://localhost:$PORT/weather-lab.html"
if lsof -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Server already running on port $PORT."
  open -a "Google Chrome" "$URL"
  exit 0
fi
( sleep 1; open -a "Google Chrome" "$URL" ) &
python3 lab_server.py --port $PORT
