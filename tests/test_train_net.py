"""Tests for the policy/value network (see docs/DESIGN.md, "Training").

Two contracts matter downstream: the policy must never leak probability onto an
illegal action (MCTS would then try to apply it), and the value head must be able
to represent the antisymmetry of a two-player game — it is trained from the point
of view of the player to move, so the same position seen from both seats must be
able to come out as ``+1`` and ``-1``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ludometer.azul.engine import ACTION_SPACE, ENCODED_SIZE, AzulState
from ludometer.train.net import (
    NetConfig,
    NetEvaluator,
    PolicyValueNet,
    load_net,
    masked_log_softmax,
    masked_policy,
    save_checkpoint,
)


def sample_states(n: int = 6, seed: int = 3) -> list[AzulState]:
    """A few positions from a random game, one per move."""
    rng = np.random.default_rng(seed)
    state = AzulState.new_game(seed=seed)
    out = [state.clone()]
    while len(out) < n and not state.is_terminal:
        legal = state.legal_actions()
        state.apply(legal[int(rng.integers(len(legal)))])
        if not state.is_terminal:
            out.append(state.clone())
    return out


@pytest.fixture(scope="module")
def net() -> PolicyValueNet:
    torch.manual_seed(0)
    return PolicyValueNet(NetConfig(hidden=64, blocks=2, value_hidden=32))


def test_config_shapes_and_defaults() -> None:
    cfg = NetConfig()
    assert cfg.input_size == ENCODED_SIZE
    assert cfg.action_space == ACTION_SPACE
    assert NetConfig.from_dict(
        {"hidden": 32, "blocks": 1, "unrelated": 9}
    ) == NetConfig(hidden=32, blocks=1)
    net = PolicyValueNet(NetConfig(hidden=16, blocks=1, value_hidden=8))
    logits, value = net(torch.zeros(4, ENCODED_SIZE))
    assert logits.shape == (4, ACTION_SPACE)
    assert value.shape == (4,)


def test_masked_policy_never_puts_probability_on_illegal_actions(
    net: PolicyValueNet,
) -> None:
    for state in sample_states():
        legal = state.legal_actions()
        logits, _ = net.evaluate_batch(state.encode()[None, :])
        row = logits[0].copy()
        # make the illegal actions overwhelmingly attractive: masking must win
        illegal = np.setdiff1d(np.arange(ACTION_SPACE), np.asarray(legal))
        row[illegal] += 50.0
        probs = masked_policy(row, legal)
        assert probs.shape == (ACTION_SPACE,)
        assert probs[illegal].sum() == 0.0
        assert probs[np.asarray(legal)].min() > 0.0
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_masked_log_softmax_matches_masked_policy(net: PolicyValueNet) -> None:
    state = sample_states(1)[0]
    legal = state.legal_actions()
    logits, _ = net.evaluate_batch(state.encode()[None, :])
    mask = torch.zeros(1, ACTION_SPACE, dtype=torch.bool)
    mask[0, torch.as_tensor(legal)] = True
    log_probs = masked_log_softmax(torch.from_numpy(logits), mask)
    probs = log_probs.exp().numpy()[0]
    assert np.all(np.isneginf(log_probs.numpy()[0][~mask.numpy()[0]]))
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(probs, masked_policy(logits[0], legal), atol=1e-5)


def test_evaluator_priors_align_with_legal_actions(net: PolicyValueNet) -> None:
    evaluator = NetEvaluator(net)
    for state in sample_states(4):
        legal = state.legal_actions()
        priors, value = evaluator(state, legal)
        assert priors.shape == (len(legal),)
        assert priors.sum() == pytest.approx(1.0, abs=1e-5)
        assert -1.0 <= value <= 1.0
        full = evaluator.full_policy(state, legal)
        assert full[np.asarray(legal)].sum() == pytest.approx(1.0, abs=1e-5)
        assert full.sum() == pytest.approx(1.0, abs=1e-5)


def test_value_head_stays_bounded(net: PolicyValueNet) -> None:
    x = torch.randn(32, ENCODED_SIZE) * 100.0
    with torch.no_grad():
        _, value = net(x)
    assert torch.isfinite(value).all()
    assert float(value.abs().max()) <= 1.0


def _swapped_encoding(state: AzulState) -> np.ndarray:
    """The same position encoded from the other seat's point of view."""
    other = state.clone()
    other.current_player = 1 - other.current_player
    return other.encode()


def test_value_head_learns_the_two_seats_with_opposite_signs() -> None:
    """Symmetry sanity: one position, both seats, targets +1 / -1.

    This is the convention the trainer relies on (value = outcome for the player
    to move), so the head plus the loss must be able to fit both signs.
    """
    torch.manual_seed(1)
    states = sample_states(8, seed=11)
    mine = np.stack([s.encode() for s in states])
    theirs = np.stack([_swapped_encoding(s) for s in states])
    x = torch.from_numpy(np.concatenate([mine, theirs]))
    y = torch.cat([torch.ones(len(mine)), -torch.ones(len(theirs))])

    net = PolicyValueNet(NetConfig(hidden=64, blocks=2, value_hidden=32))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(400):
        _, value = net(x)
        loss = torch.nn.functional.mse_loss(value, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        _, value = net(x)
    assert float(loss.detach()) < 0.05
    assert float(value[: len(mine)].min()) > 0.5
    assert float(value[len(mine) :].max()) < -0.5


def test_checkpoint_roundtrip(tmp_path, net: PolicyValueNet) -> None:
    path = save_checkpoint(tmp_path / "ckpt.pt", net, {"games": 7, "steps": 3})
    assert path.exists()
    restored, payload = load_net(path)
    assert payload["games"] == 7
    assert restored.config == net.config
    x = torch.randn(5, ENCODED_SIZE)
    with torch.no_grad():
        a = net(x)
        b = restored(x)
    torch.testing.assert_close(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])


def test_numpy_state_dict_roundtrip(net: PolicyValueNet) -> None:
    """This is what crosses the process boundary to the self-play workers."""
    weights = net.cpu_state_dict()
    assert all(isinstance(v, np.ndarray) for v in weights.values())
    other = PolicyValueNet(net.config)
    other.load_numpy_state_dict(weights)
    x = torch.randn(3, ENCODED_SIZE)
    with torch.no_grad():
        torch.testing.assert_close(net(x)[1], other(x)[1])
