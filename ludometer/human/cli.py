"""Command line for the human-games pipeline.

    python -m ludometer.human.cli endpoints            # the URL table, no network
    python -m ludometer.human.cli selftest             # synthetic round-trip, no network
    python -m ludometer.human.cli ranking  --out data/human --pages 20
    python -m ludometer.human.cli players  --out data/human --top 200 --min-elo 650
    python -m ludometer.human.cli crawl    --out data/human --cookies ~/bga_cookies.txt
    python -m ludometer.human.cli tables   --out data/human --cookies ~/bga_cookies.txt
    python -m ludometer.human.cli inspect  data/human/raw/712345678.json.gz
    python -m ludometer.human.cli convert  --out data/human
    python -m ludometer.human.cli dataset  --out data/human --npz data/human/replay.npz

``crawl`` is the one to use: it walks the all-time ladder **in rank order** (rank
1's games first), resumes exactly where a previous run stopped, and stops cleanly
when BGA's daily replay quota answers. ``tables`` is the older
whole-selection-in-one-pass loop, kept because it is what docs/HUMAN_GAMES.md
describes.

``ranking`` is the only subcommand that works without cookies. Everything that
touches a session stops on the first authentication failure rather than hammering
the login wall, and the pace defaults come from
:class:`ludometer.human.fetch.CrawlPace` (one request every 6-10 s, 120 replays a
day). Watch it happen with::

    python3 web/make_harvest.py --state data/human --watch 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ludometer.human.client import (
    AZUL_GAME_ID,
    AccountDisabled,
    AuthRequired,
    BgaClient,
    ClientConfig,
    ReplayLimitReached,
    endpoints,
    read_json_gz,
)
from ludometer.human.convert import ConversionError, convert_game
from ludometer.human.dataset import (
    OPPONENT_DROP,
    OPPONENT_VALUE_ONLY,
    DatasetStats,
    GameMeta,
    build_dataset,
    stats_path_for,
)
from ludometer.human.fetch import (
    CrawlPace,
    Fetcher,
    TableFilter,
    elo_after_map,
    select_players,
)
from ludometer.human.fixture import synthetic_log
from ludometer.human.parse import ParseError, log_type_histogram, parse_log

#: exit codes, so a cron wrapper can tell "come back tomorrow" from "stop".
EXIT_REPLAY_LIMIT = 4
EXIT_ACCOUNT_DISABLED = 5
EXIT_AUTH = 3


def _pace(args: argparse.Namespace) -> CrawlPace:
    return CrawlPace(
        min_interval=args.min_interval,
        jitter=args.jitter,
        max_requests_per_day=args.max_per_day,
        max_tables_per_day=args.max_tables_per_day,
    )


def _client(args: argparse.Namespace) -> BgaClient:
    return BgaClient(
        ClientConfig(
            cookies_path=Path(args.cookies).expanduser() if args.cookies else None,
            min_interval=args.min_interval,
            jitter=args.jitter,
            max_requests_per_day=args.max_per_day,
            max_requests_per_run=args.max_per_run,
        )
    )


def _fetcher(args: argparse.Namespace) -> Fetcher:
    return Fetcher(
        client=_client(args),
        out_dir=Path(args.out),
        table_filter=TableFilter(
            min_player_elo_raw=args.min_elo + 1300 if args.min_elo else 0.0
        ),
        game_id=AZUL_GAME_ID,
        pace=_pace(args),
    )


def cmd_endpoints(args: argparse.Namespace) -> int:
    for name, template in endpoints().items():
        print(f"{name:14s} {template}")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Prove the parse/convert/dataset chain on an engine-generated fake log."""
    ok = 0
    for seed in range(args.games):
        game, payload, infos = synthetic_log(seed=seed, swap_seats=bool(seed % 2))
        replay = parse_log(payload, 999_000_001, game.player_ids, infos=infos)
        converted = convert_game(replay)
        expected = tuple(reversed(game.scores)) if game.swap_seats else game.scores
        assert converted.scores == expected, (converted.scores, expected)
        assert len(converted) == len(game.actions)
        ok += 1
        print(
            f"seed {seed}: {len(converted)} positions, {converted.rounds} rounds, "
            f"scores {converted.scores}, outcome {converted.outcome:+.0f}"
        )
    print(f"{ok}/{args.games} synthetic games round-tripped")
    return 0


def cmd_ranking(args: argparse.Namespace) -> int:
    fetcher = _fetcher(args)
    rows = fetcher.fetch_ranking(pages=args.pages, force=args.force)
    for row in rows[: args.show]:
        print(
            f"#{row.rank:>4} {row.elo_display:>5} (raw {row.elo_raw:7.1f}) "
            f"{row.games_played:>6} games  {row.name}"
        )
    print(f"{len(rows)} players, {fetcher.client.requests_made} requests this run")
    return 0


def cmd_players(args: argparse.Namespace) -> int:
    fetcher = _fetcher(args)
    rows = fetcher.state.ranking_rows()
    if not rows:
        print("no ranking snapshot yet — run `ranking` first", file=sys.stderr)
        return 2
    kept = select_players(
        rows,
        top_n=args.top,
        min_elo_display=args.min_elo or None,
        min_games=args.min_games,
    )
    for row in kept:
        print(
            f"{row.player_id:>10} {row.elo_display:>5} {row.games_played:>6} {row.name}"
        )
    print(f"{len(kept)} players selected out of {len(rows)}")
    return 0


def cmd_tables(args: argparse.Namespace) -> int:
    fetcher = _fetcher(args)
    if not fetcher.client.authenticated:
        print(
            "no session cookie loaded: player histories and replay logs need one.\n"
            "Export cookies for boardgamearena.com to a Netscape cookies.txt and pass "
            "--cookies (see docs/HUMAN_GAMES.md).",
            file=sys.stderr,
        )
        return 2
    rows = select_players(
        fetcher.state.ranking_rows(),
        top_n=args.top,
        min_elo_display=args.min_elo or None,
        min_games=args.min_games,
    )
    # getGames/tableinfos/logs all require the per-session X-Request-Token header;
    # scrape it once before spending any authenticated request (see crawl).
    fetcher.client.fetch_request_token()
    downloaded = skipped = errors = 0
    try:
        for row in rows:
            for table_id in fetcher.fetch_player_tables(
                row.player_id, max_pages=args.history_pages
            )[: args.per_player]:
                verdict = fetcher.fetch_table(table_id)
                downloaded += verdict.status == "downloaded"
                skipped += verdict.status == "skipped"
                errors += verdict.status == "error"
                if args.limit and downloaded >= args.limit:
                    raise KeyboardInterrupt
    except ReplayLimitReached as exc:
        # BGA's per-account daily replay quota. Everything fetched so far is saved
        # and the state file remembers it, so tomorrow's run resumes for free.
        print(
            f"BGA's daily replay limit is reached: {exc}\n"
            "Stopping. Re-run in ~24 h — the state file resumes where this left off.",
            file=sys.stderr,
        )
        return 4
    except AccountDisabled as exc:
        print(
            f"BGA has disabled replay access for this account: {exc}\n"
            "Stop here and talk to BGA before retrying.",
            file=sys.stderr,
        )
        return 5
    except AuthRequired as exc:
        print(f"session rejected by BGA: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("stopping (limit reached or interrupted)")
    print(
        f"downloaded {downloaded}, skipped {skipped}, errors {errors}; "
        f"{fetcher.client.requests_made} requests this run"
    )
    return 0


def cmd_crawl(args: argparse.Namespace) -> int:
    """Walk the all-time ladder in rank order, resumably, at a conservative pace."""
    fetcher = _fetcher(args)
    if not fetcher.client.authenticated:
        print(
            "no session cookie loaded: player histories and replay logs need one.\n"
            "Export cookies for boardgamearena.com to a Netscape cookies.txt and pass "
            "--cookies (see docs/HUMAN_GAMES.md).",
            file=sys.stderr,
        )
        return 2
    rows = fetcher.state.ranking_rows()
    if not rows:
        # the ladder is public, so fetching it costs no session risk at all
        pages = max(1, (args.top + 9) // 10)
        print(f"no ranking snapshot yet — fetching the top {pages * 10} (public)")
        rows = fetcher.fetch_ranking(pages=pages)
    selected = select_players(
        rows,
        top_n=args.top,
        min_elo_display=args.min_elo or None,
        min_games=args.min_games,
    )
    if not selected:
        print("no players match the thresholds", file=sys.stderr)
        return 2
    cursor = fetcher.state.get_cursor()
    print(
        f"{len(selected)} players selected (ranks "
        f"{selected[0].rank}-{selected[-1].rank}); pace: {fetcher.pace.describe()}"
    )
    if cursor.rank:
        print(
            f"resuming at rank {cursor.rank} (player {cursor.player_id}), "
            f"table offset {cursor.table_offset}, {cursor.players_done} players done"
        )
    fetcher.client.fetch_request_token()
    report = fetcher.crawl_ranked(
        selected,
        per_player=args.per_player,
        history_pages=args.history_pages,
        max_tables=args.limit,
        resume=not args.restart,
        # --min-elo (displayed) also floors each GAME by the source player's
        # elo_after at the time it was played: a top player's beginner era
        # must not spend replay quota
        min_source_elo_raw=(args.min_elo + 1300.0) if args.min_elo else 0.0,
    )
    print(
        f"downloaded {report.downloaded}, skipped {report.skipped}, "
        f"errors {report.errors}, cached {report.cached}; "
        f"{fetcher.client.requests_made} requests this run; "
        f"cursor now rank {report.cursor.rank} offset {report.cursor.table_offset}"
    )
    if report.stopped:
        print(f"stopped: {report.stopped} — {report.message}", file=sys.stderr)
    return {
        "replay-limit": EXIT_REPLAY_LIMIT,
        "account-disabled": EXIT_ACCOUNT_DISABLED,
        "auth": EXIT_AUTH,
    }.get(report.stopped, 0)


def cmd_inspect(args: argparse.Namespace) -> int:
    """Print the notification types in a raw log — how to fill in ``LogSchema``."""
    payload = read_json_gz(Path(args.path))
    logs = payload.get("logs", payload)
    for name, row in sorted(
        log_type_histogram(logs).items(), key=lambda kv: -kv[1]["count"]
    ):
        print(f"{row['count']:>5}  {name}")
        print(f"       arg keys: {row['arg_keys']}")
        print(f"       example : {json.dumps(row['example'])[:300]}")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    fetcher = _fetcher(args)
    good = bad = 0
    reasons: dict[str, int] = {}
    for table_id, payload in fetcher.iter_raw():
        infos = payload.get("infos") or {}
        seats = ((infos.get("data") or {}).get("players") or {}).keys()
        try:
            replay = parse_log(
                payload.get("logs") or {},
                table_id,
                [int(s) for s in seats],
                infos=infos,
            )
            convert_game(replay)
            good += 1
        except (ConversionError, ParseError) as exc:
            bad += 1
            key = str(exc).split(":")[-1].strip()[:60]
            reasons[key] = reasons.get(key, 0) + 1
    print(f"{good} games convert, {bad} rejected")
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {count:>5}  {reason}")
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    """Build ``replay.npz`` from the raw payloads — **elite player's turns only**.

    Every downloaded game is replayed in full (that is the validation), but only
    the target player's decisions become training rows, with the value and margin in
    that player's frame. ``--min-target-elo`` is a floor on the *target*, in website
    units.
    """
    fetcher = _fetcher(args)
    ranking_rows = fetcher.state.ranking_rows()
    ranked_ids = [row.player_id for row in ranking_rows]
    # The ladder is the only place the target's Elo actually lives: a real
    # `tableinfos` payload carries no per-seat Elo, so `--min-target-elo` has to be
    # resolved from here or every target reads as null and the floor drops all of
    # them. `elo_raw` is BGA's raw (~1500-centred) number; the CLI floor is in
    # displayed units and is converted to raw below (displayed + 1300).
    ranked_elos = {row.player_id: row.elo_raw for row in ranking_rows}
    stats = DatasetStats()
    games: list[tuple] = []
    for table_id, payload in fetcher.iter_raw():
        infos = payload.get("infos") or {}
        seats = ((infos.get("data") or {}).get("players") or {}).keys()
        try:
            replay = parse_log(
                payload.get("logs") or {},
                table_id,
                [int(s) for s in seats],
                infos=infos,
            )
            game = convert_game(replay)
        except (ConversionError, ParseError) as exc:
            stats.note_rejected(str(exc).split(":")[-1].strip())
            continue
        games.append(
            (
                game,
                GameMeta.from_raw(
                    table_id, payload, ranked_ids, ranked_elos=ranked_elos
                ),
            )
        )
    stats = build_dataset(
        games,
        Path(args.npz),
        min_target_elo_raw=(args.min_target_elo + 1300.0)
        if args.min_target_elo
        else 0.0,
        dual_target_min_elo_raw=(args.dual_target_min_elo + 1300.0)
        if args.dual_target_min_elo
        else 0.0,
        opponent_rows=OPPONENT_VALUE_ONLY if args.keep_opponent_rows else OPPONENT_DROP,
        require_target=not args.allow_untargeted,
        stats=stats,
        # per-game Elo-at-game-time, harvested from the getGames history rows —
        # lands in the stats sidecar's game_records as `target_elo_after`
        elo_at_game=elo_after_map(fetcher.rows_path),
    )
    print(
        f"wrote {args.npz}: {stats.positions} positions from {stats.games} games "
        f"({stats.elite_positions} elite policy targets out of "
        f"{stats.all_positions} decision points; target record {stats.target_outcomes})"
    )
    if stats.games_rejected or stats.skipped_below_elo or stats.skipped_no_target:
        print(
            f"dropped: {stats.games_rejected} failed replay validation, "
            f"{stats.skipped_below_elo} below the target Elo floor, "
            f"{stats.skipped_no_target} with no identifiable elite player"
        )
    print(f"stats sidecar: {stats_path_for(args.npz)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ludometer.human.cli", description=__doc__)
    parser.add_argument(
        "--out", default="data/human", help="working directory (state + raw)"
    )
    parser.add_argument("--cookies", default=None, help="Netscape cookies.txt for BGA")
    default_pace = CrawlPace()
    parser.add_argument(
        "--min-interval",
        type=float,
        default=default_pace.min_interval,
        help="seconds between requests, before jitter (default: conservative)",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=default_pace.jitter,
        help="extra random seconds added to every gap",
    )
    parser.add_argument(
        "--max-per-day",
        type=int,
        default=default_pace.max_requests_per_day,
        help="our own cap on ALL requests per day",
    )
    parser.add_argument(
        "--max-tables-per-day",
        type=int,
        default=default_pace.max_tables_per_day,
        help="our own cap on replays per day (BGA's own quota is stricter)",
    )
    parser.add_argument("--max-per-run", type=int, default=0)
    parser.add_argument("--min-elo", type=int, default=0, help="displayed-Elo floor")
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--min-games", type=int, default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("endpoints").set_defaults(func=cmd_endpoints)

    p = sub.add_parser("selftest")
    p.add_argument("--games", type=int, default=5)
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("ranking")
    p.add_argument("--pages", type=int, default=10, help="10 players per page")
    p.add_argument("--show", type=int, default=20)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_ranking)

    sub.add_parser("players").set_defaults(func=cmd_players)

    p = sub.add_parser("crawl", help="rank-ordered, resumable, rate-limited crawl")
    p.add_argument("--per-player", type=int, default=120)
    p.add_argument("--history-pages", type=int, default=12)
    p.add_argument("--limit", type=int, default=0, help="stop after N downloads")
    p.add_argument(
        "--restart",
        action="store_true",
        help="re-walk the ladder from rank 1 instead of resuming at the cursor; "
        "already-judged tables are revisited for free (no request), so this is "
        "how earlier ranks' histories are deepened after raising --per-player / "
        "--history-pages",
    )
    p.set_defaults(func=cmd_crawl)

    p = sub.add_parser("tables")
    p.add_argument("--per-player", type=int, default=50)
    p.add_argument("--history-pages", type=int, default=5)
    p.add_argument("--limit", type=int, default=0, help="stop after N downloads")
    p.set_defaults(func=cmd_tables)

    p = sub.add_parser("inspect")
    p.add_argument("path")
    p.set_defaults(func=cmd_inspect)

    sub.add_parser("convert").set_defaults(func=cmd_convert)

    p = sub.add_parser("dataset")
    p.add_argument("--npz", default="data/human/replay.npz")
    p.add_argument(
        "--min-target-elo",
        type=int,
        default=0,
        help="displayed-Elo floor on the ELITE (target) player, not on the table",
    )
    p.add_argument(
        "--dual-target-min-elo",
        type=int,
        default=0,
        help="displayed-Elo floor above which the OPPONENT's moves also become "
        "policy targets (both seats super strong -> learn from both). 0 = off.",
    )
    p.add_argument(
        "--keep-opponent-rows",
        action="store_true",
        help="also store the opponent's positions, with policy_mask=0 "
        "(value/margin only). Off by default: learn from the elite player only.",
    )
    p.add_argument(
        "--allow-untargeted",
        action="store_true",
        help="keep games where the elite player cannot be identified (both "
        "players' moves become policy targets) — off by default",
    )
    p.set_defaults(func=cmd_dataset)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
