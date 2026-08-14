"""Flask server for the play-vs-AI GUI (``uv run ludometer-gui``).

One game lives in process memory (this is a local single-player toy, not a
service). The page in ``web/play/`` is served from disk and talks to:

======================  ===========================================================
``POST /api/new``       ``{opponent_spec, human_plays_first, seed?}`` -> full state
``GET  /api/state``     full state, legal actions, AI's last move, game log
``POST /api/act``       ``{action_id}`` -> human move + AI reply + new state
``GET  /api/hint``      the heuristic agent's suggestion for the human's turn
``GET  /api/agents``    the agent specs the dropdown offers
======================  ===========================================================

Every error is a JSON ``{"error": "..."}`` with a 400/404/500 status — the page
turns those into a toast, so a broken ``mcts:`` checkpoint spec never kills the
running game.
"""

from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from ludometer.gui.session import GameSession, IllegalMove

__all__ = ["BASELINE_SPECS", "create_app", "main"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8737
DEFAULT_SPEC = "heuristic"
BASELINE_SPECS = ("heuristic", "greedy", "random")

# repo layout: <root>/ludometer/gui/server.py and <root>/web/play/
PLAY_DIR = Path(__file__).resolve().parents[2] / "web" / "play"


def create_app(play_dir: Path | None = None) -> Flask:
    """Build the Flask app. ``play_dir`` overrides where the page is read from."""
    static_dir = Path(play_dir) if play_dir is not None else PLAY_DIR
    app = Flask(__name__, static_folder=None)
    lock = threading.Lock()
    box: dict[str, GameSession | None] = {"session": None}

    # ------------------------------------------------------------------ errors
    @app.errorhandler(IllegalMove)
    def _illegal(exc: IllegalMove):  # pragma: no cover - exercised via routes
        return jsonify({"error": str(exc)}), 400

    @app.errorhandler(HTTPException)
    def _http(exc: HTTPException):
        return jsonify({"error": exc.description, "status": exc.code}), exc.code or 500

    @app.errorhandler(Exception)
    def _boom(exc: Exception):  # pragma: no cover - last-resort net
        app.logger.exception("unhandled error")
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    def fail(message: str, status: int = 400):
        return jsonify({"error": message}), status

    def current() -> GameSession | None:
        return box["session"]

    # -------------------------------------------------------------------- page
    @app.get("/")
    def index():
        if not (static_dir / "index.html").is_file():
            return fail(f"page not found: {static_dir / 'index.html'}", 500)
        return send_from_directory(static_dir, "index.html")

    @app.get("/<path:filename>")
    def asset(filename: str):
        if not (static_dir / filename).is_file():
            return fail(f"no such file: {filename}", 404)
        return send_from_directory(static_dir, filename)

    # --------------------------------------------------------------------- api
    @app.get("/api/agents")
    def agents():
        return jsonify(
            {
                "baselines": list(BASELINE_SPECS),
                "default": DEFAULT_SPEC,
                "custom_example": "mcts:runs/<run>/checkpoints/<name>.pt?sims=400",
            }
        )

    @app.post("/api/new")
    def new_game():
        payload: Any = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return fail("body must be a JSON object")
        spec = payload.get("opponent_spec", DEFAULT_SPEC)
        if not isinstance(spec, str) or not spec.strip():
            return fail("opponent_spec must be a non-empty string")
        spec = spec.strip()
        human_first = payload.get("human_plays_first", True)
        if not isinstance(human_first, bool):
            return fail("human_plays_first must be true or false")
        seed = payload.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError):
                return fail("seed must be an integer")
        try:
            session = GameSession(spec, human_plays_first=human_first, seed=seed)
        except Exception as exc:  # noqa: BLE001 - bad spec / missing ckpt / no torch
            return fail(
                f"could not load opponent {spec!r}: {type(exc).__name__}: {exc}"
            )
        with lock:
            box["session"] = session
            return jsonify(session.snapshot())

    @app.get("/api/state")
    def state():
        with lock:
            session = current()
            if session is None:
                return fail("no game in progress — POST /api/new first", 409)
            return jsonify(session.snapshot())

    @app.post("/api/act")
    def act():
        payload: Any = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return fail("body must be a JSON object")
        raw = payload.get("action_id")
        if isinstance(raw, bool) or not isinstance(raw, int):
            try:
                raw = int(raw)  # accept "42" from a form-ish client
            except (TypeError, ValueError):
                return fail(f"action_id must be an integer, got {raw!r}")
        with lock:
            session = current()
            if session is None:
                return fail("no game in progress — POST /api/new first", 409)
            try:
                return jsonify(session.play_human(int(raw)))
            except IllegalMove as exc:
                return fail(str(exc))
            except ValueError as exc:  # engine-level rejection
                return fail(str(exc))

    @app.get("/api/hint")
    def hint():
        with lock:
            session = current()
            if session is None:
                return fail("no game in progress — POST /api/new first", 409)
            try:
                return jsonify(session.hint())
            except IllegalMove as exc:
                return fail(str(exc))
            except Exception as exc:  # noqa: BLE001 - pragma: no cover
                return fail(f"hint failed: {type(exc).__name__}: {exc}", 500)

    return app


def main(argv: list[str] | None = None) -> int:
    """``ludometer-gui`` entry point: serve the page and open a browser tab."""
    parser = argparse.ArgumentParser(description="Play Azul against a Ludometer agent")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open a browser tab"
    )
    parser.add_argument("--debug", action="store_true", help="Flask debug reloader")
    args = parser.parse_args(argv)

    app = create_app()
    url = f"http://{args.host}:{args.port}/"
    print(f"Ludometer GUI on {url}  (Ctrl-C to stop)")
    if not args.no_browser and not args.debug:
        threading.Timer(0.7, webbrowser.open, args=(url,)).start()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
