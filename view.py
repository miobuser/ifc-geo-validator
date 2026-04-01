"""Quick viewer — validate an IFC file and open the 3D viewer.

Usage:
    python view.py tests/test_models/T8_curved_wall.ifc
    python view.py tests/test_models/T4_l_shaped.ifc
    python view.py                                        # starts server for drag & drop
"""
import sys
import os
import json
import webbrowser
import threading
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

sys.path.insert(0, str(Path(__file__).parent / "src"))

VIEWER_DIR = Path(__file__).parent / "viewer"
PORT = 8080


def export_model(ifc_path):
    """Run validation pipeline and export JSON for viewer."""
    from viewer.export_for_viewer import export_model as _export
    return _export(ifc_path, str(VIEWER_DIR / "model_data.json"))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(VIEWER_DIR), **kw)

    def do_POST(self):
        if self.path == "/validate":
            import tempfile
            length = int(self.headers["Content-Length"])
            body = self.rfile.read(length)
            with tempfile.NamedTemporaryFile(suffix=".ifc", delete=False) as f:
                f.write(body)
                tmp = f.name
            try:
                from viewer.server import run_pipeline
                result = run_pipeline(tmp)
                data = json.dumps(result)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data.encode())
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            finally:
                os.unlink(tmp)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt, *args):
        if "POST" in str(args) or "404" in str(args):
            super().log_message(fmt, *args)


def main():
    # If IFC path given, export first
    if len(sys.argv) > 1 and sys.argv[1].endswith(".ifc"):
        ifc_path = sys.argv[1]
        if not os.path.exists(ifc_path):
            print(f"File not found: {ifc_path}")
            sys.exit(1)
        export_model(ifc_path)

    # Start server
    print(f"\n>>> Server: http://localhost:{PORT}/viewer.html")
    print(f">>> Drag & drop IFC files onto the page to validate")
    print(f">>> Ctrl+C to stop\n")

    server = HTTPServer(("localhost", PORT), Handler)

    # Open browser
    webbrowser.open(f"http://localhost:{PORT}/viewer.html")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
