"""Rate-limited, cookie-aware HTTP client for boardgamearena.com.

Everything this package sends to BGA goes through :class:`BgaClient`, which owns
the three things we promised to get right (see ``docs/HUMAN_GAMES.md``):

* **one request at a time, slowly** — a minimum interval between requests with a
  little jitter, plus a hard per-run and per-day cap. There is no concurrency
  anywhere in this package, on purpose;
* **a real browser identity** — a desktop Chrome ``User-Agent`` and the
  ``X-Requested-With``/``Referer`` pair BGA's own XHRs send, so the traffic is
  shaped like a logged-in tab rather than like a crawler;
* **cookies from a file Remi exports himself** — the ranking endpoint is public,
  but player histories and replay logs are not (BGA answers them with
  ``{"status":"0", ..., "code":806}``). We never log in programmatically and we
  never touch the password: the client reads a Netscape ``cookies.txt`` (what
  every "export cookies" browser extension writes) or a raw ``Cookie:`` header
  string, and stops with :class:`AuthRequired` the moment BGA says the session is
  not valid.

Standard library only (``urllib`` + ``http.cookiejar``): the project depends on
numpy/torch/flask and this package must not add ``requests`` to that list.
"""

from __future__ import annotations

import gzip
import http.cookiejar
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "AZUL_GAME_ID",
    "BGA_HOST",
    "AccountDisabled",
    "AuthRequired",
    "BgaClient",
    "BgaError",
    "ClientConfig",
    "RateLimitExceeded",
    "ReplayLimitReached",
    "ReplayUnavailable",
    "display_elo",
    "endpoints",
    "raw_elo",
    "read_json_gz",
    "write_json_gz",
]

#: BGA's numeric id for Azul (read out of the public game list on ``/gamepanel``).
#: ``azulduel`` is 2220, ``azulsummerpavilion`` 1911, ``azulqueensgarden`` 2560 —
#: different games, not options of this one.
AZUL_GAME_ID = 1467

#: The English subdomain. The apex ``boardgamearena.com`` 302s here, so asking for
#: it directly halves the request count, and the ranking widget's mode labels are
#: only in English on this host.
BGA_HOST = "https://en.boardgamearena.com"

#: A current desktop Chrome UA. BGA serves the anonymous ranking JSON to anything,
#: but a default ``Python-urllib/3.x`` UA is exactly the fingerprint that gets
#: rate-limiters interested.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)


class BgaError(RuntimeError):
    """Any BGA-side failure (HTTP error, or ``status: 0`` in the JSON body)."""


class AuthRequired(BgaError):
    """BGA refused for lack of a session — code 806, or a redirect to /account.

    Raised as its own type because it is not retryable and not a rate problem:
    the fetcher stops the whole run and tells the user which cookies to export.
    """


class RateLimitExceeded(BgaError):
    """Our own budget, not BGA's: the run or the day hit its request cap."""


class ReplayLimitReached(BgaError):
    """BGA's **per-account daily replay quota** is spent. Stop until tomorrow.

    This is the single most important operational fact about collecting BGA
    replays, and it is not an HTTP 429 — it is a normal 200 with
    ``error: "You have reached a limit (replay)"`` in the JSON. BGA caps how many
    archived games one account may open per day because serving them is expensive
    (see ``docs/HUMAN_GAMES.md`` §7), and the cap resets roughly 24 h after it is
    hit. Other projects work around it by rotating several accounts; **we do not** —
    that is evasion, and one account's cap is the honest budget.
    """


class AccountDisabled(BgaError):
    """BGA has disabled replay access for this account. Stop and tell the user."""


class ReplayUnavailable(BgaError):
    """This particular game's archive is gone — skip the table, keep going."""


def endpoints() -> dict[str, str]:
    """The BGA URL templates this package knows about, as a documented table.

    Returned as data (rather than hidden in call sites) so that
    ``python -m ludometer.human.cli endpoints`` can print the same list the
    handoff document describes, and a future reader can diff the two.

    ``auth`` is ``"public"`` for the one endpoint that answers anonymously and
    ``"session"`` for everything that needs Remi's cookies — verified live on
    2026-08-17, see ``docs/HUMAN_GAMES.md``.
    """
    return {
        # public: 10 rows per call, mode=elo is the ALL-TIME ladder (mode=arena is
        # the current season, which is a different number entirely).
        "ranking": "/gamepanel/gamepanel/getRanking.html?game={game}&start={start}&mode=elo",
        # session + X-Request-Token: a player's finished games for one game id.
        # Parameter names taken from working community code (DavidEGx/bga-duel-finder):
        # game_id, player, opponent_id, updateStats, start_date, end_date. Rows carry
        # `table_id`, `scores` and `players` as comma-joined strings, which is why the
        # `table_infos` call below may turn out to be skippable — see docs §5.2.
        "player_tables": "/gamestats/gamestats/getGames.html"
        "?player={player}&game_id={game}&opponent_id=0&finished=1&updateStats=0&page={page}",
        # session: BGA appears to need this before it will serve the log — three
        # independent projects call it first and one comments "seemingly required to
        # produce log". It is what counts against the daily replay quota.
        "archive_prime": "/gamereview/gamereview/requestTableArchive.html?table={table}",
        # session: the move log of one finished table — the actual replay data.
        # `translated=true` is what working community code sends; `false` returns the
        # untranslated log templates and is equally parseable (we read `args`, never
        # the `log` sentence).
        "table_logs": "/archive/archive/logs.html?table={table}&translated=true",
        # session: table metadata, including the game options (variant, speed...).
        # NOTE robots.txt disallows /table, so prefer `table_infos_alt`, which is the
        # same payload from a path robots.txt does not mention.
        "table_infos": "/table/table/tableinfos.html?id={table}",
        "table_infos_alt": "/tablemanager/tablemanager/tableinfos.html?id={table}",
        # session, HTML: the human-facing replay page. Its `g_gamelogs` global holds
        # the same log JSON (see `parse.parse_gamelogs_html`), which makes a
        # browser-saved copy of this page a zero-request route to a real replay.
        # `{version}` is the game version string, e.g. "260626-1038" for Azul today.
        "replay_page": "/archive/replay/{version}/?table={table}&player={player}&comments=",
    }


@dataclass
class ClientConfig:
    """Politeness budget. The defaults are the ones the report recommends.

    ``min_interval`` 3 s with ``jitter`` 1.5 s averages ~3.75 s between requests,
    i.e. ~16/minute and ~23k/day if it ran flat out for 24 h — well under the
    ``max_requests_per_day`` cap, which is the real limit. Both are deliberately
    slower than a human clicking through replays, because a human does not do it
    for eight hours straight.
    """

    min_interval: float = 3.0
    jitter: float = 1.5
    timeout: float = 30.0
    max_retries: int = 3
    retry_backoff: float = 15.0
    max_requests_per_day: int = 4000
    max_requests_per_run: int = 0  # 0 = no per-run cap
    host: str = BGA_HOST
    user_agent: str = USER_AGENT
    cookies_path: Path | None = None
    cookie_header: str | None = None
    dry_run: bool = False
    seed: int = 0


@dataclass
class BgaClient:
    """One-at-a-time GET client. Create once per run and share it."""

    config: ClientConfig = field(default_factory=ClientConfig)
    requests_made: int = 0
    last_request: float = 0.0
    #: `bgaConfig.requestToken`, sent as `X-Request-Token` once known. Fill it with
    #: :meth:`fetch_request_token` before the first `getGames` call.
    request_token: str | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.config.seed)
        self._jar = http.cookiejar.MozillaCookieJar()
        path = self.config.cookies_path
        if path is not None:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"cookie file not found: {p}")
            # ignore_discard/ignore_expires: browser exports routinely mark the
            # session cookie as a session-only cookie, which the strict reader
            # would then drop and leave us silently anonymous.
            self._jar.load(str(p), ignore_discard=True, ignore_expires=True)
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self._jar),
            _NoRedirectToLogin(),
        )

    # ------------------------------------------------------------------ cookies
    @property
    def authenticated(self) -> bool:
        """True when a session cookie is present (not that it is still valid)."""
        if self.config.cookie_header:
            return "PHPSESSID" in self.config.cookie_header
        return any(c.name in _SESSION_COOKIES for c in self._jar)

    def cookie_names(self) -> list[str]:
        """Names of the cookies we loaded — handy for "did the export work?"."""
        return sorted({c.name for c in self._jar})

    # ------------------------------------------------------------------ fetching
    def _wait(self) -> None:
        gap = self.config.min_interval + self._rng.random() * self.config.jitter
        remaining = gap - (time.monotonic() - self.last_request)
        if self.last_request and remaining > 0:
            time.sleep(remaining)

    def _check_budget(self) -> None:
        cap = self.config.max_requests_per_run
        if cap and self.requests_made >= cap:
            raise RateLimitExceeded(f"per-run cap reached ({cap} requests)")

    def get(self, path: str, referer: str | None = None) -> tuple[int, bytes]:
        """One raw GET against :attr:`config.host`. Returns ``(status, body)``.

        Retries only on 5xx and network errors, with a linear backoff; a 4xx is
        returned to the caller, which knows what it means.
        """
        self._check_budget()
        url = path if path.startswith("http") else self.config.host + path
        if self.config.dry_run:
            return 0, b""
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "en-US,en;q=0.9",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": referer or f"{self.config.host}/gamepanel?game=azul",
        }
        if self.config.cookie_header:
            headers["Cookie"] = self.config.cookie_header
        if self.request_token:
            # Some authenticated AJAX endpoints (notably gamestats/getGames) want the
            # per-session CSRF token the site puts in `bgaConfig.requestToken`; BGA's
            # own JS sends it as this header. See `fetch_request_token`.
            headers["X-Request-Token"] = self.request_token
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            self._wait()
            request = urllib.request.Request(url, headers=headers)
            self.requests_made += 1
            self.last_request = time.monotonic()
            try:
                with self._opener.open(
                    request, timeout=self.config.timeout
                ) as response:
                    return int(response.status), response.read()
            except urllib.error.HTTPError as exc:
                body = exc.read()
                if exc.code < 500:
                    return int(exc.code), body
                last_error = exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
            time.sleep(self.config.retry_backoff * (attempt + 1))
        raise BgaError(
            f"GET {url} failed after {self.config.max_retries} tries: {last_error}"
        )

    def get_json(self, path: str, referer: str | None = None) -> dict[str, Any]:
        """GET a BGA AJAX endpoint and unwrap its envelope.

        BGA answers every AJAX call with ``{"status": 1, "data": ...}`` on success
        and ``{"status": "0", "error": ..., "code": 806}`` on failure; a
        session-less call to a private endpoint is that 806, and a session-less
        call to a *page* is a 302 to ``/account?warn&redirect=...`` instead. Both
        become :class:`AuthRequired` here so callers never have to look.
        """
        status, body = self.get(path, referer=referer)
        if status == 0 and self.config.dry_run:
            return {}
        if status in (301, 302, 303, 307, 308):
            raise AuthRequired(f"{path} redirected ({status}) — session required")
        text = body.decode("utf-8", "replace")
        if text.lstrip().startswith("<"):
            raise AuthRequired(f"{path} answered HTML, not JSON — session required")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BgaError(f"{path}: response is not JSON ({exc})") from exc
        if str(payload.get("status")) not in ("1", "True"):
            raise _classify(path, payload)
        # A 200 with status 1 can still carry an error field on the archive path.
        error = str(payload.get("error") or "")
        if error:
            raise _classify(path, payload)
        return payload

    # -------------------------------------------------------------- convenience
    def fetch_request_token(self, path: str = "/gamepanel?game=azul") -> str | None:
        """Scrape ``bgaConfig.requestToken`` off any BGA page and remember it.

        Costs one request. BGA embeds a per-session CSRF token in every page as
        ``requestToken: '<hex>'`` inside the ``bgaConfig`` literal, and its own JS
        replays it as the ``X-Request-Token`` header on some authenticated AJAX
        calls — ``gamestats/getGames.html`` among them, per working community code
        (``DavidEGx/bga-duel-finder``). Anonymous pages carry a token too, so this
        works before login, but the token that matters is one fetched **with** the
        cookies.
        """
        _status, body = self.get(path)
        match = re.search(
            r"requestToken:\s*'([0-9a-f]{16,128})'", body.decode("utf-8", "replace")
        )
        self.request_token = match.group(1) if match else None
        return self.request_token

    def ranking_page(
        self, start: int, game: int = AZUL_GAME_ID
    ) -> list[dict[str, Any]]:
        """One page of 10 rows of the **all-time** ladder. Public, no cookies.

        Each row is BGA's own dict: ``id`` (player id, string), ``name``,
        ``ranking`` (the *raw* Elo, e.g. ``"2486.16"``), ``nbr_game``,
        ``rank_no``, ``country``, ``avatar``. The number the website shows is
        ``max(0, raw - 1300)`` floored — see :func:`display_elo`.
        """
        path = endpoints()["ranking"].format(game=game, start=int(start))
        payload = self.get_json(path)
        ranks = (payload.get("data") or {}).get("ranks") or []
        return list(ranks)


#: BGA's own error strings, quoted from the projects that hit them in production
#: (``rhstephens/hivemind``, ``liamdj/tokaido-analysis``, ``HStrand/bga-tm-scraper``).
#: They arrive inside a 200 response, so they have to be matched on text.
REPLAY_LIMIT_TEXT = "you have reached a limit (replay)"
ACCOUNT_DISABLED_TEXT = "disabled for your account"
EMPTY_ARCHIVE_TEXT = "replay for this game has been lost"
NEW_ACCOUNT_TEXT = "registered more than 24 hours"


def _classify(path: str, payload: dict[str, Any]) -> BgaError:
    """Turn a BGA error envelope into the most specific exception we have."""
    code = payload.get("code")
    error = str(payload.get("error", ""))
    low = error.lower()
    if REPLAY_LIMIT_TEXT in low:
        return ReplayLimitReached(f"{path}: {error}")
    if ACCOUNT_DISABLED_TEXT in low:
        return AccountDisabled(f"{path}: {error}")
    if EMPTY_ARCHIVE_TEXT in low:
        return ReplayUnavailable(f"{path}: {error}")
    if NEW_ACCOUNT_TEXT in low:
        # The account is too new/inactive for archive access; not retryable.
        return AuthRequired(f"{path}: {error}")
    if str(code) == "806" or "session" in low:
        return AuthRequired(f"{path}: {error} (code {code})")
    return BgaError(f"{path}: {error} (code {code})")


#: Cookie names BGA uses for a logged-in session. ``PHPSESSID`` is the session
#: itself; the ``TournoiEnLigne*`` pair is the "remember me" token that lets the
#: server re-create a session, so exporting all three survives a session timeout.
_SESSION_COOKIES = frozenset(
    {"PHPSESSID", "TournoiEnLigneid", "TournoiEnLigneidt", "TournoiEnLignelang"}
)


class _NoRedirectToLogin(urllib.request.HTTPRedirectHandler):
    """Return the 302 instead of following it to ``/account``.

    A session-less request for a *page* (``/gamestats?...``) is answered with a
    redirect to the login wall, and the wall is a 1.8 MB HTML page. Following it
    would cost a second request and tell us nothing, so we surface the redirect.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if "/account" in newurl:
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def display_elo(raw: float | str) -> int:
    """The Elo the website shows, from the ``ranking`` field the API returns.

    BGA stores a classic ~1500-centred Elo and displays ``max(0, elo - 1300)``
    floored (the site's own JS: ``Math.max(0, parseFloat(e) - 1300)``). Remi's
    older CSV dump holds *displayed* numbers, this API holds *raw* ones, and
    mixing them up is a 1300-point mistake — hence this function.
    """
    return max(0, int(float(raw) - 1300))


def raw_elo(display: float) -> float:
    """Inverse of :func:`display_elo`, for turning a threshold into an API value."""
    return float(display) + 1300.0


def write_json_gz(path: Path, payload: Any) -> Path:
    """Store a fetched payload compressed — replay logs are ~10x smaller gzipped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    tmp.replace(path)
    return path


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)
