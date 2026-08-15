"""Build the two tiny ONNX graphs margin.test.mjs runs against.

Neither is a net: each one just *routes* its input to its outputs, so a test can
say exactly what the "net" should answer for a given observation vector and then
assert on it. That is the point — what is under test is the player's plumbing
(does it notice a third output? does it read the right row of a batch?), not any
learned function.

    obs : float32[batch, 182]
      -> policy = obs[:, 0:180]     (logits, one per action)
      -> value  = obs[:, 180:181]
      -> margin = obs[:, 181:182]   (three-output graph only)

`toy_two_output.onnx` stops at value, and stands in for every model exported
before the margin head existed — the player must keep its old behaviour there.

Regenerate with:

    uv run --group export python web/player/test/fixtures/make_toy_onnx.py
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
OBS = 182
ACTIONS = 180


def const(name: str, values: list[int]):
    return numpy_helper.from_array(np.array(values, dtype=np.int64), name)


def slice_node(out: str, start: int, end: int, tag: str):
    """out = obs[:, start:end] — Slice with its bounds as initializers."""
    return helper.make_node(
        "Slice",
        inputs=["obs", f"{tag}_start", f"{tag}_end", f"{tag}_axis"],
        outputs=[out],
        name=f"slice_{tag}",
    )


def build(with_margin: bool, path: Path) -> None:
    cuts = [("policy", 0, ACTIONS), ("value", ACTIONS, ACTIONS + 1)]
    if with_margin:
        cuts.append(("margin", ACTIONS + 1, ACTIONS + 2))

    nodes, inits, outputs = [], [], []
    for name, start, end in cuts:
        nodes.append(slice_node(name, start, end, name))
        inits += [
            const(f"{name}_start", [start]),
            const(f"{name}_end", [end]),
            const(f"{name}_axis", [1]),
        ]
        outputs.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", end - start])
        )

    graph = helper.make_graph(
        nodes,
        "toy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["batch", OBS])],
        outputs,
        initializer=inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="ludometer-toy"
    )
    model.ir_version = 9  # what onnxruntime-web 1.27 accepts
    onnx.checker.check_model(model)
    path.write_bytes(model.SerializeToString())
    print(f"wrote {path} ({path.stat().st_size} bytes), outputs: {[o.name for o in outputs]}")


if __name__ == "__main__":
    build(True, HERE / "toy_margin.onnx")
    build(False, HERE / "toy_two_output.onnx")
