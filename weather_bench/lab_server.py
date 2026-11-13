"""Local server for the weather lab page. Started by start-lab.command.

Does two jobs:
  1. Serves this folder on http://localhost:8077, like `python3 -m http.server`.
     Web Serial only works from localhost, not from a file opened in Finder.
  2. Accepts POST /save?name=... and writes the body into runs/. That is what
     lets the page autosave its data to disk without a download dialog, so a
     teammate or a script can read the run straight from runs/.

Only plain file names ending in .json or .csv are accepted, only into runs/,
and only from a page served by this same server (checked with the Origin
header), so no other website open in Chrome can write files here.

    python3 lab_server.py            # port 8077
    python3 lab_server.py --port 8078
"""
import argparse
import http.server
import json
import os
import re
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
NAME_OK = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._\-]{0,118}\.(json|csv)")
MAX_BYTES = 60 * 1024 * 1024  # a 6-hour 2 Hz session is about 3 MB; this is headroom


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)

    def end_headers(self):
        # always hand Chrome the current page, never a cached copy from before an edit
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def reply(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        if url.path != "/save":
            return self.reply(404, {"error": "only /save"})
        port = self.server.server_address[1]
        origin = self.headers.get("Origin", "")
        if origin not in (f"http://localhost:{port}", f"http://127.0.0.1:{port}"):
            return self.reply(403, {"error": "wrong origin"})
        name = urllib.parse.parse_qs(url.query).get("name", [""])[0]
        if not NAME_OK.fullmatch(name):
            return self.reply(400, {"error": "bad file name"})
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BYTES:
            return self.reply(413, {"error": "empty or too large"})
        data = self.rfile.read(n)
        os.makedirs(RUNS, exist_ok=True)
        path = os.path.join(RUNS, name)
        # write to a side file, then swap it in, so a crash mid-write never leaves half a file
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        self.reply(200, {"saved": "runs/" + name, "bytes": n})

    def log_message(self, fmt, *args):
        # keep the Terminal window readable: only show saves and errors
        msg = fmt % args
        if "POST" in msg or " 4" in msg or " 5" in msg:
            super().log_message(fmt, *args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8077)
    args = ap.parse_args()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Serving {HERE} at http://localhost:{args.port}/weather-lab.html")
    print("Saves land in runs/. Leave this window open while testing. Ctrl-C stops it.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
