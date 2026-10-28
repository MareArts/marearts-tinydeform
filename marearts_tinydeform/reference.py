# Copyright 2026 MareArts
# SPDX-License-Identifier: Apache-2.0
"""The standard path, in plain PyTorch: project every memory token, then sample with grid_sample.

This is the computation used by Deformable-DETR-style decoders (RT-DETR, D-FINE, DINO and others) and the
reference the fused kernel is tested against. Runs on CPU and GPU and supports autograd.
"""

import torch
import torch.nn.functional as F


def ma_deform_attn_reference(memory, value_weight, value_bias, spatial_shapes, level_start_index,
                             sampling_locations, attention_weights):
    """value = memory @ value_weight.T + value_bias, then multi-scale deformable attention.

    memory              [B, N, C]
    value_weight        [H * D, C]      (an nn.Linear weight)
    value_bias          [H * D] or None
    spatial_shapes      [L, 2] int64    (height, width) of each level, tokens laid out level by level
    level_start_index   [L] int64       first token of each level
    sampling_locations  [B, Q, H, L, P, 2]  (x, y) normalised to [0, 1]
    attention_weights   [B, Q, H, L, P]     already normalised (softmax over L * P)
    returns             [B, Q, H * D]
    """
    B, N, _ = memory.shape
    _, Q, H, L, P, _ = sampling_locations.shape
    value = F.linear(memory, value_weight, value_bias).view(B, N, H, -1)
    D = value.shape[-1]
    grids = 2 * sampling_locations - 1
    samples = []
    for lvl in range(L):
        h, w = int(spatial_shapes[lvl, 0]), int(spatial_shapes[lvl, 1])
        s = int(level_start_index[lvl])
        v = value[:, s:s + h * w].flatten(2).transpose(1, 2).reshape(B * H, D, h, w)
        g = grids[:, :, :, lvl].transpose(1, 2).flatten(0, 1)  # [B * H, Q, P, 2]
        samples.append(F.grid_sample(v, g, mode='bilinear', padding_mode='zeros', align_corners=False))
    a = attention_weights.transpose(1, 2).reshape(B * H, 1, Q, L * P)
    out = (torch.stack(samples, dim=-2).flatten(-2) * a).sum(-1)  # [B * H, D, Q]
    return out.view(B, H * D, Q).transpose(1, 2).contiguous()
