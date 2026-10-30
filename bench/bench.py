# Copyright 2026 MareArts
# SPDX-License-Identifier: Apache-2.0
"""Time the decoder's value projection + multi-scale deformable attention, three ways, on one GPU:

  pytorch   nn.Linear on every memory token + grid_sample per level (the plain PyTorch path)
  official  nn.Linear on every memory token + the Deformable-DETR CUDA kernel (only if the module
            MultiScaleDeformableAttention is importable — build it from the Deformable-DETR repository)
  fused     marearts_tinydeform.ma_deform_attn(mode='fused')    sample memory rows first, project per (query, head)
  project   marearts_tinydeform.ma_deform_attn(mode='project')  nn.Linear on every token + marearts-tinydeform's sampling kernel
  auto      marearts_tinydeform.ma_deform_attn() (the default)  picks fused or project by size

Each case is captured in a CUDA graph and replayed, so Python launch overhead is not counted. Times in µs.
Shapes default to RT-DETR at 640x640: levels 80x80, 40x40, 20x20 (8,400 tokens), C=256, 8 heads, 4 points.

  python bench/bench.py [--queries 50 100 300] [--batch 1 4] [--dtype float32 float16]
"""

import argparse
import sys

import torch
import torch.nn.functional as F

from marearts_tinydeform import choose_mode, ma_deform_attn, ma_deform_attn_reference

try:
    import MultiScaleDeformableAttention as MSDA
except ImportError:
    MSDA = None


def official(memory, weight, bias, shapes, starts, loc, attn):
    B, N, _ = memory.shape
    H = loc.shape[2]
    value = F.linear(memory, weight, bias).view(B, N, H, -1)
    out = MSDA.ms_deform_attn_forward(value, shapes, starts, loc, attn, 64)
    return out  # [B, Q, H * D]


def timed_us(fn, reps=100):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = float('inf')
    for _ in range(7):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1e3 / reps)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queries', type=int, nargs='+', default=[50, 100, 300])
    ap.add_argument('--batch', type=int, nargs='+', default=[1, 4])
    ap.add_argument('--dtype', nargs='+', default=['float32', 'float16'])
    ap.add_argument('--levels', default='80x80,40x40,20x20')
    ap.add_argument('--channels', type=int, default=256)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--points', type=int, default=4)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')  # the table has '—'; a Windows console or pipe is often not UTF-8
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device('cuda')
    levels = [tuple(int(v) for v in s.split('x')) for s in args.levels.split(',')]
    shapes = torch.tensor(levels, dtype=torch.long, device=dev)
    starts = torch.cat([shapes.new_zeros(1), (shapes[:, 0] * shapes[:, 1]).cumsum(0)[:-1]])
    N, C, H, L, P = int((shapes[:, 0] * shapes[:, 1]).sum()), args.channels, args.heads, len(levels), args.points

    print(f'{torch.cuda.get_device_name()}, torch {torch.version.__version__} CUDA {torch.version.cuda}, '
          f'tokens {N}, C {C}, heads {H}, levels {L}, points {P}, TF32 off')
    print('| dtype | batch | queries | pytorch | official | fused | project | auto (picks) | auto vs pytorch |'
          ' auto vs official | max abs diff vs pytorch |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for name in args.dtype:
        dtype = getattr(torch, name)
        for B in args.batch:
            for Q in args.queries:
                g = torch.Generator(device=dev).manual_seed(0)
                memory = torch.randn(B, N, C, device=dev, generator=g).to(dtype)
                weight = (torch.randn(C, C, device=dev, generator=g) / C ** 0.5).to(dtype)
                bias = (torch.randn(C, device=dev, generator=g) * 0.1).to(dtype)
                loc = (torch.rand(B, Q, 1, 1, 1, 2, device=dev, generator=g) +
                       0.05 * torch.randn(B, Q, H, L, P, 2, device=dev, generator=g)).to(dtype)
                attn = torch.randn(B, Q, H, L * P, device=dev, generator=g).softmax(-1).view(B, Q, H, L, P).to(dtype)
                a = (memory, weight, bias, shapes, starts, loc, attn)
                # the PyTorch path reads level sizes on the host (a GPU read would sync inside graph capture)
                a_ref = (memory, weight, bias, shapes.cpu(), starts.cpu(), loc, attn)
                with torch.inference_mode():
                    ref = ma_deform_attn_reference(*a_ref)
                    diff = max((ma_deform_attn(*a, mode=m).float() - ref.float()).abs().max().item()
                               for m in ('fused', 'project'))
                    t_ref = timed_us(lambda: ma_deform_attn_reference(*a_ref))
                    t_off = timed_us(lambda: official(*a)) if MSDA is not None else None
                    t_fus = timed_us(lambda: ma_deform_attn(*a, mode='fused'))
                    t_prj = timed_us(lambda: ma_deform_attn(*a, mode='project'))
                pick = choose_mode(memory, weight, loc)
                t_auto = t_fus if pick == 'fused' else t_prj
                off = f'{t_off:.1f}' if t_off else '—'
                vs_off = f'{t_off / t_auto:.1f}x' if t_off else '—'
                print(f'| {name} | {B} | {Q} | {t_ref:.1f} | {off} | {t_fus:.1f} | {t_prj:.1f} | {t_auto:.1f} ({pick}) |'
                      f' {t_ref / t_auto:.1f}x | {vs_off} | {diff:.1e} |')


if __name__ == '__main__':
    main()
