"""Tests for the play-vs-AI web GUI (see docs/DESIGN.md and ludometer/gui/).

Flask test client only — no browser. What matters here is that the API never
lies to the page: legal action ids are playable, illegal input is a clean 400
instead of a traceback, and the round-end reports the overlays render agree with
the engine's own scoring.
"""

from __future__ import annotations

import pytest

from ludometer.agents.registry import load_agent
from ludometer.azul.engine import CENTER, FLOOR, AzulState, encode_action
from ludometer.gui.moves import describe_action, final_report, round_report
from ludometer.gui.server import PLAY_DIR, create_app
from ludometer.gui.session import GameSession

MAX_PLIES = 400  # a 2-player game is ~40 moves; this is only a runaway guard


@pytest.fixture
def client():
    app = create_app()
    app.testing = True
    with app.test_client() as c:
        yield c


def start(client, **kwargs):
    body = {"opponent_spec": "heuristic", "human_plays_first": True, "seed": 12}
    body.update(kwargs)
    response = client.post("/api/new", json=body)
    assert response.status_code == 200, response.get_json()
    return response.get_json()


# ------------------------------------------------------------------ page files
def test_page_is_served(client):
    assert (PLAY_DIR / "index.html").is_file()
    page = client.get("/")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "app.js" in html and "style.css" in html
    for asset in ("app.js", "style.css"):
        r = client.get("/" + asset)
        assert r.status_code == 200, asset
        assert r.get_data()
    assert client.get("/nope.js").status_code == 404


def test_page_has_no_external_resources():
    """The GUI must work offline: no CDN, no remote fonts or images."""
    for name in ("index.html", "style.css", "app.js"):
        text = (PLAY_DIR / name).read_text()
        for needle in ("http://", "https://", "//cdn", "cdnjs", "googleapis"):
            assert needle not in text, f"{name} references {needle}"


def test_missing_page_directory_is_a_clean_error(tmp_path):
    app = create_app(play_dir=tmp_path)
    app.testing = True
    response = app.test_client().get("/")
    assert response.status_code == 500
    assert "page not found" in response.get_json()["error"]


# ------------------------------------------------------------------- new / state
def test_state_needs_a_game_first(client):
    for path in ("/api/state", "/api/hint"):
        response = client.get(path)
        assert response.status_code == 409
        assert "no game in progress" in response.get_json()["error"]
    response = client.post("/api/act", json={"action_id": 0})
    assert response.status_code == 409


def test_new_game_state_shape(client):
    data = start(client)
    for key in (
        "state",
        "seed",
        "opponent_spec",
        "agent_name",
        "opponent_info",
        "opponent_blurb",
        "human_seat",
        "ai_seat",
        "your_turn",
        "legal_actions",
        "human_legal_actions",
        "log",
        "final",
        "last_ai_move",
        "last_round_report",
        "render_text",
        "ply",
    ):
        assert key in data, key

    state = data["state"]
    assert state["round"] == 0
    assert state["current_player"] == 0
    assert len(state["factories"]) == 5
    assert all(len(f) == 5 for f in state["factories"])
    assert sum(sum(f) for f in state["factories"]) == 20
    assert state["center"] == [0] * 5
    assert state["marker_in_center"] is True
    assert sum(state["bag"]) == 80
    assert state["lid"] == [0] * 5
    assert state["tiles_left"] == 20
    assert state["is_terminal"] is False
    assert state["scores"] == [0, 0]
    assert len(state["players"]) == 2
    for player in state["players"]:
        assert [row["capacity"] for row in player["pattern_lines"]] == [1, 2, 3, 4, 5]
        assert player["wall"] == [[0] * 5 for _ in range(5)]
        assert player["floor"] == [0] * 5
        assert player["floor_penalty"] == 0

    assert data["human_seat"] == 0 and data["ai_seat"] == 1
    assert data["your_turn"] is True
    assert data["seed"] == 12
    assert data["agent_name"] == "heuristic"
    assert data["opponent_info"] is None and data["opponent_blurb"] is None
    assert data["final"] is None
    assert data["last_ai_move"] is None
    assert data["human_legal_actions"] == data["legal_actions"]
    assert data["human_legal_actions"], "an opening position always has moves"
    assert all(0 <= a < 180 for a in data["human_legal_actions"])
    assert data["log"] and data["log"][0]["kind"] == "start"


def test_ai_opens_when_the_human_takes_the_second_seat(client):
    data = start(client, human_plays_first=False)
    assert data["human_seat"] == 1 and data["ai_seat"] == 0
    assert data["your_turn"] is True  # the AI already replied
    assert data["state"]["current_player"] == 1
    assert data["last_ai_move"] is not None
    assert data["last_ai_move"]["player"] == 0
    assert data["ply"] == 1
    assert data["human_legal_actions"]


def test_seed_is_optional_and_reproducible(client):
    a = start(client, seed=None)
    assert isinstance(a["seed"], int)
    b = start(client, seed=99)
    c = start(client, seed=99)
    assert b["state"]["factories"] == c["state"]["factories"]


def test_agents_endpoint_lists_the_dropdown(client):
    """The dropdown offers the baselines plus the auto-resolved best checkpoint.

    ``default`` is "best" on a machine that has trained something and
    "heuristic" on a fresh clone — see tests/test_registry_best.py for both paths
    against a fake runs/ tree.
    """
    data = client.get("/api/agents").get_json()
    assert data["default"] in ("best", "heuristic")
    assert data["fallback_default"] == "heuristic"
    assert set(data["baselines"]) == {"heuristic", "greedy", "random"}
    assert "mcts:" in data["custom_example"]
    best = data["best"]
    assert best["spec"] == "best" and best["label"]
    assert best["sims_choices"] and best["default_sims"] in best["sims_choices"]
    if best["available"]:
        assert data["default"] == "best"
        assert best["checkpoint"] and best["run"] and "Elo" in best["detail"]
    else:
        assert data["default"] == "heuristic" and best["error"]


# ------------------------------------------------------------------ full games
@pytest.mark.parametrize("spec", ["random", "greedy", "heuristic"])
def test_a_full_game_can_be_played_through_the_api(client, spec):
    data = start(client, opponent_spec=spec, seed=hash(spec) % 1000)
    plies = 0
    rounds_reported = 0
    while not data["state"]["is_terminal"]:
        legal = data["human_legal_actions"]
        assert legal, "the human must always have a move when it is their turn"
        assert data["your_turn"] is True
        response = client.post("/api/act", json={"action_id": legal[0]})
        assert response.status_code == 200, response.get_json()
        data = response.get_json()
        assert data["human_move"]["side"] == "human"
        for move in data["ai_moves"]:
            assert move["side"] == "ai"
            assert move["player"] == data["ai_seat"]
        rounds_reported += len(data["round_reports"])
        plies += 1
        assert plies < MAX_PLIES

    final = data["final"]
    assert final is not None
    assert final["winner"] in (0, 1, None)
    assert final["scores"] == data["state"]["scores"]
    assert rounds_reported == len(
        data["log"] and [e for e in data["log"] if e["kind"] == "round"]
    )
    assert rounds_reported >= 2
    assert data["human_legal_actions"] == []
    assert data["log"][-1]["kind"] == "end"
    # the game is closed: no more moves accepted
    assert client.post("/api/act", json={"action_id": 0}).status_code == 400
    assert client.get("/api/hint").status_code == 400


def test_round_reports_match_the_engine(client):
    """Every overlay number must be the engine's, not a re-derivation of it."""
    data = start(client, opponent_spec="greedy", seed=5)
    checked = 0
    while not data["state"]["is_terminal"]:
        response = client.post(
            "/api/act", json={"action_id": data["human_legal_actions"][0]}
        )
        data = response.get_json()
        for report in data["round_reports"]:
            assert len(report["players"]) == 2
            for seat, player in enumerate(report["players"]):
                gain = player["tiling_points"] + player["floor"]["penalty"]
                assert player["score_after"] == max(0, player["score_before"] + gain)
                assert player["delta"] == player["score_after"] - player["score_before"]
                assert (
                    sum(t["points"] for t in player["tiles"]) == player["tiling_points"]
                )
                if not report["game_over"]:
                    # mid-game the engine's score is exactly the tiled score
                    assert report["scores_after"][seat] == player["score_after"]
                else:
                    bonus = data["final"]["bonuses"][seat]["total"]
                    assert report["scores_after"][seat] == player["score_after"] + bonus
                for t in player["tiles"]:
                    assert t["col"] == (t["color"] + t["row"]) % 5
                    assert t["points"] >= 1
                checked += 1
    assert checked >= 4


def test_round_reports_agree_with_the_engine_over_many_games():
    """Fuzz: the reconstructed tiling must match the engine's score every round.

    This caught the report being built from the position *before* the round's last
    move, which silently mis-attributed the tiles that move completed.
    """
    checked = 0
    for seed in range(8):
        session = GameSession("random", human_plays_first=seed % 2 == 0, seed=seed)
        while not session.state.is_terminal:
            legal = session.legal_for_human()
            payload = session.play_human(legal[seed % len(legal)])
            for report in payload["round_reports"]:
                for seat, player in enumerate(report["players"]):
                    expected = report["scores_after"][seat]
                    if report["game_over"]:
                        expected -= payload["final"]["bonuses"][seat]["total"]
                    assert player["score_after"] == expected, (
                        seed,
                        report["round"],
                        seat,
                    )
                    checked += 1
    assert checked >= 40


# --------------------------------------------------------------- illegal input
def test_illegal_actions_are_rejected(client):
    data = start(client)
    legal = set(data["human_legal_actions"])
    illegal = next(a for a in range(180) if a not in legal)
    response = client.post("/api/act", json={"action_id": illegal})
    assert response.status_code == 400
    assert "not legal" in response.get_json()["error"]
    # the game is untouched
    assert client.get("/api/state").get_json()["ply"] == 0

    for bad in (-1, 180, 10**9, "seven", None, [3], 1.5, True):
        response = client.post("/api/act", json={"action_id": bad})
        assert response.status_code == 400, bad
        assert response.get_json()["error"]
    assert client.post("/api/act", json={}).status_code == 400
    assert (
        client.post(
            "/api/act", data="not json", content_type="application/json"
        ).status_code
        == 400
    )
    assert client.get("/api/state").get_json()["ply"] == 0


def test_bad_opponent_specs_fail_cleanly(client):
    for spec in ("bogus", "", "   ", "mcts:", 42, ["random"]):
        response = client.post("/api/new", json={"opponent_spec": spec})
        assert response.status_code == 400, spec
        assert response.get_json()["error"]

    # a checkpoint spec that cannot be loaded (missing file, or no trainer yet)
    response = client.post("/api/new", json={"opponent_spec": "mcts:nope.pt?sims=8"})
    assert response.status_code == 400
    assert "could not load opponent" in response.get_json()["error"]

    response = client.post(
        "/api/new", json={"opponent_spec": "random", "human_plays_first": "yes"}
    )
    assert response.status_code == 400
    response = client.post("/api/new", json={"opponent_spec": "random", "seed": "abc"})
    assert response.status_code == 400
    assert client.get("/api/state").status_code == 409  # nothing was started


def test_a_failed_new_game_leaves_the_running_game_alone(client):
    start(client)
    client.post(
        "/api/act",
        json={
            "action_id": client.get("/api/state").get_json()["human_legal_actions"][0]
        },
    )
    before = client.get("/api/state").get_json()
    assert (
        client.post("/api/new", json={"opponent_spec": "mcts:ghost.pt"}).status_code
        == 400
    )
    after = client.get("/api/state").get_json()
    assert after["ply"] == before["ply"]
    assert after["state"] == before["state"]


# ----------------------------------------------------------------------- hints
def test_hint_returns_a_legal_action(client):
    data = start(client)
    for _ in range(6):
        hint = client.get("/api/hint")
        assert hint.status_code == 200
        payload = hint.get_json()
        assert payload["action_id"] in data["human_legal_actions"]
        assert payload["text"].startswith("Try: ")
        move = payload["move"]
        assert move["count"] >= 1
        assert (
            0 <= move["source"] <= 5
            and 0 <= move["color"] <= 4
            and 0 <= move["dest"] <= 5
        )
        # taking the hint is always accepted
        response = client.post("/api/act", json={"action_id": payload["action_id"]})
        assert response.status_code == 200, response.get_json()
        data = response.get_json()
        if data["state"]["is_terminal"]:
            break


# ------------------------------------------------------- move descriptions
def test_describe_action_spells_out_the_move():
    state = AzulState.new_game(seed=3)
    color = next(c for c in range(5) if state.factories[0][c])
    count = state.factories[0][color]
    move = describe_action(state, encode_action(0, color, 4))
    assert move["count"] == count
    assert move["placed"] == count and move["overflow"] == 0
    assert move["took_marker"] is False
    assert move["source_label"] == "factory 1"
    assert move["dest_label"] == "row 5"
    assert move["text"].startswith(f"took {count} ")
    assert state.to_json() == AzulState.new_game(seed=3).to_json()  # no mutation

    # dumping into row 0 (capacity 1) spills the rest onto the floor
    spill = describe_action(state, encode_action(0, color, 0))
    assert spill["placed"] == 1
    assert spill["overflow"] == count - 1
    assert spill["to_floor"] == count - 1 and spill["to_lid"] == 0
    if count > 1:
        assert "to the floor" in spill["text"]

    floor = describe_action(state, encode_action(0, color, FLOOR))
    assert floor["placed"] == 0 and floor["overflow"] == count
    assert floor["dest_label"] == "the floor line"


def test_describe_action_reports_the_first_player_marker():
    state = AzulState.new_game(seed=3)
    color = next(c for c in range(5) if state.factories[0][c])
    state.apply(encode_action(0, color, FLOOR))  # pushes the rest into the center
    center_color = next(c for c in range(5) if state.center[c])
    move = describe_action(state, encode_action(CENTER, center_color, 4))
    assert move["took_marker"] is True
    assert move["source_label"] == "the center"
    assert "first-player marker" in move["text"]


def test_round_and_final_reports_on_a_played_out_game():
    state = AzulState.new_game(seed=8)
    agent = load_agent("greedy", seed=1)
    reports = []
    while not state.is_terminal:
        before = state.clone()
        round_before = state.round_index
        action = agent.act(state)
        move = describe_action(state, action)
        state.apply(action)
        if state.round_index != round_before or state.is_terminal:
            reports.append((before, move, round_report(before, move)))
    assert len(reports) >= 3
    for before, move, report in reports:
        assert report["round"] == before.round_index
        for seat, player in enumerate(report["players"]):
            assert player["seat"] == seat
            assert player["score_before"] == before.scores[seat]
            if seat != move["player"]:  # the mover's floor changed with the move
                assert player["floor"]["penalty"] == before.floor_penalty(seat)

    final = final_report(state, human_seat=0)
    assert final["scores"] == state.scores
    for seat, bonus in enumerate(final["bonuses"]):
        assert bonus["rows"] == state.completed_rows(seat)
        assert bonus["cols"] == state.completed_cols(seat)
        assert bonus["colors"] == state.completed_colors(seat)
        assert (
            bonus["total"]
            == 2 * bonus["rows"] + 7 * bonus["cols"] + 10 * bonus["colors"]
        )
        assert bonus["score_before_bonus"] + bonus["total"] == state.scores[seat]
    assert final_report(AzulState.new_game(seed=0), human_seat=0) is None


def test_session_narrates_and_never_mutates_out_of_turn():
    session = GameSession("random", human_plays_first=True, seed=4)
    payload = session.play_human(session.legal_for_human()[0])
    assert payload["human_move"]["label"] == "You"
    assert payload["ai_moves"][0]["label"] == "AI"
    assert payload["ply"] == 2
    texts = [entry["text"] for entry in payload["log"]]
    assert any(t.startswith("You took") for t in texts)
    assert any(t.startswith("AI took") for t in texts)
    with pytest.raises(ValueError):
        session.play_human(-5)


# -------------------------------------------------------------- agent registry
def test_registry_builds_the_baselines():
    for spec in ("random", "greedy", "heuristic"):
        assert load_agent(spec, seed=1).name == spec
        assert load_agent(" " + spec + " ").name == spec
    assert load_agent("random").name == "random"


def test_registry_rejects_bad_specs():
    for spec in (
        "",
        "nope",
        "mcts:",
        "mcts:x?sims=abc",
        "mcts:x?sims=0",
        "mcts:x?depth=2",
    ):
        with pytest.raises(ValueError):
            load_agent(spec)
    with pytest.raises(TypeError):
        load_agent(7)  # type: ignore[arg-type]
