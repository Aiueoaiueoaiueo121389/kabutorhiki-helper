import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "dist"


class SpaHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        requested = ROOT / self.path.lstrip("/").split("?", 1)[0]
        if not requested.exists() or requested.is_dir():
            self.path = "/index.html"
        return super().do_GET()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    ThreadingHTTPServer(("0.0.0.0", port), SpaHandler).serve_forever()
