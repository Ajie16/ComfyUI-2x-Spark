"""Regression: INT8 ConvRot qkv_proj scales must follow the Ulysses head split.

Kitchen int8_linear does `weight_scale.expand(n)` when numel != n. A sharded
qkv with qdata rows N/world but scale still [N, 1] raises:

    expand(FloatTensor{[21504, 1]}, size=[10752])

Run with ComfyUI's venv from the ComfyUI root:
    .venv/bin/python custom_nodes/buqi-minimax-h3-multigpu/tests/test_head_shard_int8.py
"""
from __future__ import annotations

import os
import sys

import torch

PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.dirname(os.path.dirname(PLUGIN))
sys.path.insert(0, COMFY)
sys.path.insert(0, PLUGIN)

from comfy_kitchen.tensor.base import QuantizedTensor  # noqa: E402
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout  # noqa: E402

from minimax_sp.sp_forward import (  # noqa: E402
    _qkv_head_index,
    head_sharded_qkv,
)


class _Proj:
    def __init__(self, weight, bias=None):
        self.weight = weight
        self.bias = bias
        self._parameters = {"weight": weight}
        self.out_features = int(weight.shape[0])


def _int8_weight(rows, cols, convrot=True):
    qdata = torch.arange(rows * cols, dtype=torch.int8).reshape(rows, cols)
    scale = (torch.arange(1, rows + 1, dtype=torch.float32) * 0.01).reshape(rows, 1)
    params = TensorWiseINT8Layout.Params(
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=(rows, cols),
        is_weight=True,
        convrot=convrot,
        convrot_groupsize=256,
    )
    return QuantizedTensor(qdata, "TensorWiseINT8Layout", params)


def test_qkv_index_matches_h3_world2():
    idx = _qkv_head_index(21504, 2, 0, torch.device("cpu"))
    assert idx.numel() == 10752
    inner = 7168
    assert idx[:3584].tolist() == list(range(0, 3584))
    assert idx[3584:7168].tolist() == list(range(inner, inner + 3584))
    assert idx[7168:].tolist() == list(range(2 * inner, 2 * inner + 3584))


def test_int8_rowwise_scale_is_sliced():
    rows, cols, world, rank = 12, 8, 2, 0
    proj = _Proj(_int8_weight(rows, cols))
    sub = head_sharded_qkv(proj, world, rank)
    assert sub.out_features == rows // world
    assert tuple(sub.weight._qdata.shape) == (rows // world, cols)
    scale = sub.weight._params.scale
    assert tuple(scale.shape) == (rows // world, 1)
    idx = _qkv_head_index(rows, world, rank, torch.device("cpu"))
    torch.testing.assert_close(scale, proj.weight._params.scale[idx])
    torch.testing.assert_close(sub.weight._qdata, proj.weight._qdata[idx])
    # kitchen: numel == n skips expand([N,1] -> n)
    assert scale.numel() == sub.weight.shape[0]


def test_per_tensor_scale_is_left_alone():
    rows, cols = 12, 8
    qdata = torch.zeros(rows, cols, dtype=torch.int8)
    params = TensorWiseINT8Layout.Params(
        scale=torch.tensor(0.02, dtype=torch.float32),
        orig_dtype=torch.bfloat16,
        orig_shape=(rows, cols),
        is_weight=True,
        convrot=False,
        convrot_groupsize=256,
    )
    proj = _Proj(QuantizedTensor(qdata, "TensorWiseINT8Layout", params))
    sub = head_sharded_qkv(proj, 2, 1)
    assert sub.weight._params.scale.numel() == 1
    torch.testing.assert_close(sub.weight._params.scale.cpu().float(),
                               torch.tensor(0.02, dtype=torch.float32))
    assert sub.weight._qdata.shape[0] == rows // 2


def test_bias_is_sliced():
    rows, cols = 12, 8
    bias = torch.arange(rows, dtype=torch.float32)
    proj = _Proj(torch.randn(rows, cols), bias=bias)
    sub = head_sharded_qkv(proj, 2, 0)
    idx = _qkv_head_index(rows, 2, 0, torch.device("cpu"))
    torch.testing.assert_close(sub.bias, bias[idx])


def test_kitchen_expand_matches_sharded_rows():
    """The exact int8_linear line that crashed on unsliced [N,1] scales."""
    unsliced = torch.zeros(21504, 1)
    try:
        unsliced.expand(10752)
    except RuntimeError as exc:
        assert "expand" in str(exc)
    else:
        raise AssertionError("unsliced [N,1].expand(N/2) should fail")

    proj = _Proj(_int8_weight(12, 8))
    sub = head_sharded_qkv(proj, 2, 0)
    n = sub.weight.shape[0]
    ws = sub.weight._params.scale
    if ws.numel() != n:
        ws.expand(n).contiguous()
    assert ws.numel() == n


if __name__ == "__main__":
    test_qkv_index_matches_h3_world2()
    test_int8_rowwise_scale_is_sliced()
    test_per_tensor_scale_is_left_alone()
    test_bias_is_sliced()
    test_kitchen_expand_matches_sharded_rows()
    print("ok")
