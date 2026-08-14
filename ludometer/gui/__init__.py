"""Human-vs-AI web GUI: Flask API (:mod:`ludometer.gui.server`) + `web/play/`.

The browser talks to a single in-memory :class:`~ludometer.gui.session.GameSession`
over four JSON endpoints; all Azul rules live in :mod:`ludometer.azul.engine`, and
this package only *describes* what the engine did so the page can narrate it.
"""

from __future__ import annotations

__all__ = ["GameSession", "create_app", "main"]


def __getattr__(name: str):  # lazy so `import ludometer.gui` stays flask-free
    if name in ("create_app", "main"):
        from ludometer.gui import server

        return getattr(server, name)
    if name == "GameSession":
        from ludometer.gui.session import GameSession

        return GameSession
    raise AttributeError(name)
