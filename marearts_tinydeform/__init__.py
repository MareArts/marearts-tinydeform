# Copyright 2026 MareArts
# SPDX-License-Identifier: Apache-2.0
"""marearts-tinydeform: value projection + multi-scale deformable attention for DETR-style decoders (CUDA, inference)."""

import functools

import torch
import torch.nn.functional as F

from .reference import ma_deform_attn_reference

__all__ = ['ma_deform_attn', 'ma_deform_attn_reference', 'choose_mode']
__version__ = '0.1.1'

try:
    from . import _C
except ImportError:  # built without CUDA, or not built: the CUDA path raises, the reference still works
    _C = None

MODES = ('auto', 'fused', 'project')

# 'fused' costs ~ batch * queries * samples * C (row reads); 'project' costs ~ batch * tokens * C * H * D (a GEMM)
# plus a fixed part that matters at batch 1. So auto picks 'fused' while
#     queries * levels * points * 4 / (tokens * head_dim)  <  a + b / batch
# with (a, b) fitted to the crossovers measured with bench/bench.py (queries 50 ... 900, batch 1 ... 8):
# FUSED_LIMIT on an RTX 4090 (sm_89), used from sm_80 on; FUSED_LIMIT_SM75 on an RTX 2080 Super (sm_75), used
# below sm_80, where the GEMM in 'project' is slower relative to sampling (bfloat16 has no tensor cores there).
# Other GPUs may cross over elsewhere: pass mode= to override.
FUSED_LIMIT = {torch.float32: (0.073, 0.032), torch.float16: (0.023, 0.031), torch.bfloat16: (0.023, 0.031)}
FUSED_LIMIT_SM75 = {torch.float32: (0.068, 0.0), torch.float16: (0.020, 0.0), torch.bfloat16: (0.132, 0.0)}


@functools.lru_cache(maxsize=None)
def _limits(device_index):
    major, _ = torch.cuda.get_device_capability(device_index)
    return FUSED_LIMIT if major >= 8 else FUSED_LIMIT_SM75


def choose_mode(memory, value_weight, sampling_locations):
    """The mode 'auto' runs: 'fused' for few queries per image, 'project' (project every token first) for many."""
    B, N, _ = memory.shape
    _, Q, H, L, P, _ = sampling_locations.shape
    D = value_weight.shape[0] // H
    limits = _limits(memory.get_device()) if memory.is_cuda else FUSED_LIMIT
    a, b = limits.get(memory.dtype, limits[torch.float16])
    return 'fused' if Q * L * P * 4 / (N * D) < a + b / max(B, 1) else 'project'


def ma_deform_attn(memory, value_weight, value_bias, spatial_shapes, level_start_index,
                   sampling_locations, attention_weights, mode='auto'):
    """value = memory @ value_weight.T + value_bias, then multi-scale deformable attention; [B, Q, H * D].

    Same arguments, shapes and result as ma_deform_attn_reference (grid_sample, bilinear, zero padding,
    align_corners=False). CUDA tensors (float32, float16, bfloat16; float32 accumulation):
      mode='fused'    sample the memory rows first, then project per (query, head) -- no projection of all tokens
      mode='project'  project all tokens (F.linear), then sample -- the usual order, our sampling kernel
      mode='auto'     choose_mode() picks one by size (default)
    CPU tensors run the reference. Inference only: inputs that require grad are rejected.
    """
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}, got {mode!r}')
    if not memory.is_cuda:
        return ma_deform_attn_reference(memory, value_weight, value_bias, spatial_shapes, level_start_index,
                                        sampling_locations, attention_weights)
    if _C is None:
        raise RuntimeError('marearts-tinydeform was installed without its CUDA extension')
    if torch.is_grad_enabled() and any(
            t is not None and t.requires_grad
            for t in (memory, value_weight, value_bias, sampling_locations, attention_weights)):
        raise RuntimeError('ma_deform_attn is inference only: run it under torch.no_grad() / '
                           'torch.inference_mode(), or use ma_deform_attn_reference for training')
    if mode == 'auto':
        mode = choose_mode(memory, value_weight, sampling_locations)
    if mode == 'fused':
        return _C.fused_forward(memory, value_weight, value_bias, spatial_shapes, level_start_index,
                                sampling_locations, attention_weights)
    value = F.linear(memory, value_weight, value_bias)
    return _C.sample_forward(value, sampling_locations.shape[2], spatial_shapes, level_start_index,
                             sampling_locations, attention_weights)
