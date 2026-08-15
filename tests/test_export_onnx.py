"""ONNX export: does the graph still answer exactly what torch answers?

The browser player is only as trustworthy as this file. A checkpoint is exported
to ONNX and both runtimes are asked the same real positions; anything past 1e-4
means the page would be playing a slightly different net from the one whose Elo
we quote.

`onnx` and `onnxruntime` are not project dependencies (only the export path needs
them), so the tests that need them skip cleanly when they are absent.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ludometer.azul.engine import ACTION_SPACE, ENCODED_SIZE, AzulState
from ludometer.export.onnx_export import (
    ExportWrapper,
    collect_observations,
    export_checkpoint,
)
from ludometer.train.net import NetConfig, PolicyValueNet, save_checkpoint

onnx = pytest.importorskip("onnx", reason="onnx is only needed to export")
ort = pytest.importorskip("onnxruntime", reason="onnxruntime is only needed to export")


@pytest.fixture(scope="module")
def small_checkpoint(tmp_path_factory) -> str:
    """A tiny but real net, saved the way the trainer saves one."""
    torch.manual_seed(0)
    net = PolicyValueNet(NetConfig(hidden=64, blocks=2, value_hidden=16))
    path = tmp_path_factory.mktemp("ckpt") / "ckpt-000042.pt"
    save_checkpoint(path, net, {"games": 42})
    return str(path)


def test_collect_observations_are_real_encodings():
    obs = collect_observations(20, seed=3)
    assert obs.shape == (20, ENCODED_SIZE)
    assert obs.dtype == np.float32
    # real positions, not noise: walls are 0/1 and the round marker is in [0, 1]
    assert set(np.unique(obs[:, :50]).tolist()) <= {0.0, 1.0}
    assert (obs[:, 175] >= 0).all() and (obs[:, 175] <= 1).all()
    # and they are not all the same position
    assert len(np.unique(obs, axis=0)) > 1


def test_export_wrapper_keeps_value_two_dimensional():
    net = PolicyValueNet(NetConfig(hidden=32, blocks=1, value_hidden=8))
    wrapper = ExportWrapper(net).eval()
    logits, value = wrapper(torch.zeros(3, ENCODED_SIZE))
    assert logits.shape == (3, ACTION_SPACE)
    assert value.shape == (3, 1)


def test_export_matches_torch(tmp_path, small_checkpoint):
    meta = export_checkpoint(
        ckpt=small_checkpoint,
        out_dir=tmp_path / "model",
        samples=40,
        reference=tmp_path / "reference.json.gz",
    )
    onnx_path = tmp_path / "model" / "model.onnx"
    assert onnx_path.is_file()
    assert meta["checkpoint"] == "ckpt-000042"
    assert meta["input_size"] == ENCODED_SIZE
    assert meta["action_space"] == ACTION_SPACE
    assert meta["games"] == 42

    parity = meta["parity"]
    assert parity["checked"] is True
    assert parity["n"] == 40
    assert parity["policy_max_abs_diff"] < 1e-4
    assert parity["value_max_abs_diff"] < 1e-4

    # the metadata the page reads must be on disk and self-consistent
    written = json.loads((tmp_path / "model" / "model_meta.json").read_text())
    assert written["onnx_bytes"] == onnx_path.stat().st_size
    assert written["num_params"] > 0


def test_exported_graph_signature_and_batching(tmp_path, small_checkpoint):
    """One input, two outputs, and a batch axis that is genuinely dynamic."""
    export_checkpoint(
        ckpt=small_checkpoint, out_dir=tmp_path / "model", samples=5, reference=None
    )
    session = ort.InferenceSession(
        str(tmp_path / "model" / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    assert [i.name for i in session.get_inputs()] == ["obs"]
    assert [o.name for o in session.get_outputs()] == ["policy", "value"]

    for batch in (1, 7):
        obs = collect_observations(batch, seed=batch)
        policy, value = session.run(None, {"obs": obs})
        assert policy.shape == (batch, ACTION_SPACE)
        assert value.shape == (batch, 1)
        assert np.isfinite(policy).all()
        assert (np.abs(value) <= 1.0).all()  # the value head is a tanh


def test_export_refuses_to_overwrite_on_a_parity_failure(
    tmp_path, small_checkpoint, monkeypatch
):
    """A bad export must not replace a good model.onnx that is already there."""
    out = tmp_path / "model"
    export_checkpoint(ckpt=small_checkpoint, out_dir=out, samples=5, reference=None)
    good = (out / "model.onnx").read_bytes()

    import ludometer.export.onnx_export as mod

    monkeypatch.setattr(
        mod,
        "verify_parity",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("nope")),
    )
    with pytest.raises(AssertionError):
        export_checkpoint(ckpt=small_checkpoint, out_dir=out, samples=5, reference=None)
    assert (out / "model.onnx").read_bytes() == good


def test_encoding_contract_the_js_port_relies_on():
    """The exported input size is the engine's, not a number typed twice."""
    state = AzulState.new_game(seed=5)
    assert state.encode().shape == (ENCODED_SIZE,)
    assert PolicyValueNet().config.input_size == ENCODED_SIZE
    assert PolicyValueNet().config.action_space == ACTION_SPACE
