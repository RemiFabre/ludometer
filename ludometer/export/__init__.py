"""Export trained checkpoints to portable formats (currently ONNX for the web).

The browser player under ``web/player/`` does not ship torch: it runs the same
policy+value net through onnxruntime-web. :mod:`ludometer.export.onnx_export`
is what turns a training checkpoint into the ``model.onnx`` that page loads.
"""

from __future__ import annotations

__all__ = ["onnx_export"]
