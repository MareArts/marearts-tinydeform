# Copyright 2026 MareArts
# SPDX-License-Identifier: Apache-2.0
"""Both CUDA paths (fused, project) vs the grid_sample reference. Run: python -m pytest tests  (or python tests/test_correctness.py)"""

import torch

import marearts_tinydeform
from marearts_tinydeform import choose_mode, ma_deform_attn, ma_deform_attn_reference

CUDA = torch.cuda.is_available()


def make(B, Q, H, D, C, levels, P, dtype=torch.float32, spread=1.0, bias=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    shapes = torch.tensor(levels, dtype=torch.long)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), (shapes[:, 0] * shapes[:, 1]).cumsum(0)[:-1]])
    N = int((shapes[:, 0] * shapes[:, 1]).sum())
    L = len(levels)
    memory = torch.randn(B, N, C, generator=g)
    weight = torch.randn(H * D, C, generator=g) / C ** 0.5
    b = torch.randn(H * D, generator=g) * 0.1 if bias else None
    # centre in [0, 1] plus offsets that also land outside the map
    loc = (torch.rand(B, Q, 1, 1, 1, 2, generator=g) + spread * 0.3 * torch.randn(B, Q, H, L, P, 2, generator=g))
    attn = torch.randn(B, Q, H, L * P, generator=g).softmax(-1).view(B, Q, H, L, P)
    return [t if t is None or t.dtype == torch.long else t.to(dtype) for t in
            (memory, weight, b, shapes, starts, loc, attn)]


def to(args, device):
    return [None if t is None else t.to(device) for t in args]


CASES = [
    # name, B, Q, H, D, C, levels, P
    ('rtdetr-640', 1, 300, 8, 32, 256, [(80, 80), (40, 40), (20, 20)], 4),
    ('rtdetr-640 b4 few queries', 4, 40, 8, 32, 256, [(80, 80), (40, 40), (20, 20)], 4),
    ('deformable-detr 4 levels non-square', 2, 100, 8, 32, 256, [(100, 152), (50, 76), (25, 38), (13, 19)], 4),
    ('narrow heads', 2, 50, 4, 16, 64, [(32, 48), (16, 24)], 8),
    ('one level many points', 1, 64, 2, 64, 128, [(20, 30)], 16),
    ('wide channels', 1, 40, 12, 64, 768, [(40, 40), (20, 20), (10, 10)], 4),
    ('odd channels (scalar kernel)', 2, 33, 3, 10, 30, [(13, 17), (7, 9)], 4),
    ('channels % 8 != 0, % 4 == 0', 1, 25, 4, 9, 36, [(11, 12)], 4),
]


def check(name, args, ref_args, atol, rtol):
    with torch.inference_mode():
        ref = ma_deform_attn_reference(*ref_args).float().cpu()
        for mode in ('fused', 'project', 'auto'):
            got = ma_deform_attn(*args, mode=mode).float().cpu()
            err = (got - ref).abs().max().item()
            assert torch.allclose(got, ref, atol=atol, rtol=rtol), \
                f'{name} [{mode}]: max |diff| {err:.3g} (max |ref| {ref.abs().max().item():.3g})'


def test_cpu_path_is_reference():
    args = make(1, 10, 2, 8, 16, [(6, 7), (3, 4)], 2)
    with torch.inference_mode():
        assert torch.equal(ma_deform_attn(*args), ma_deform_attn_reference(*args))


def test_cuda_float32():
    if not CUDA:
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for name, *shape in CASES:
        for bias in (True, False):
            args = to(make(*shape, bias=bias), 'cuda')
            check(f'{name} fp32 bias={bias}', args, args, atol=2e-5, rtol=1e-4)


def test_cuda_half_and_bfloat16():
    if not CUDA:
        return
    for dtype, atol in ((torch.float16, 3e-3), (torch.bfloat16, 2e-2)):
        for name, *shape in CASES:
            args = make(*shape, dtype=dtype)
            ref = [t if t is None or t.dtype == torch.long else t.double() for t in args]  # reference in float64
            check(f'{name} {dtype}', to(args, 'cuda'), ref, atol=atol, rtol=atol)


def test_unaligned_memory_uses_scalar_kernel():
    if not CUDA:
        return
    for dtype, atol in ((torch.float32, 2e-5), (torch.float16, 3e-3)):
        args = make(1, 40, 8, 32, 256, [(20, 20), (10, 10)], 4, dtype=dtype)
        mem = args[0].cuda()
        buf = torch.empty(mem.numel() + 1, dtype=dtype, device='cuda')
        shifted = buf[1:].view(mem.shape)  # contiguous, but 2 or 4 bytes off a 16-byte boundary
        shifted.copy_(mem)
        assert shifted.is_contiguous() and shifted.data_ptr() % 16 != 0
        dev = to(args, 'cuda')
        ref = [t if t is None or t.dtype == torch.long else t.double() for t in args]
        check(f'unaligned {dtype}', [shifted] + dev[1:], ref, atol=atol, rtol=atol)


def test_far_outside_and_empty():
    if not CUDA:
        return
    args = to(make(2, 20, 4, 8, 32, [(9, 11), (5, 6)], 4, spread=200.0), 'cuda')  # nearly all samples outside
    check('far outside', args, args, atol=1e-5, rtol=1e-4)
    empty = to(make(1, 0, 4, 8, 32, [(9, 11)], 4), 'cuda')
    with torch.inference_mode():
        for mode in ('fused', 'project'):
            assert ma_deform_attn(*empty, mode=mode).shape == (1, 0, 32)


def test_rejects_bad_input():
    if not CUDA:
        return
    args = to(make(1, 8, 2, 8, 16, [(6, 7), (3, 4)], 2), 'cuda')
    bad = [
        ('weight cols', 1, args[1][:, :8].contiguous()),
        ('dtype', 1, args[1].half()),
        ('bias size', 2, args[2][:4]),
        ('shapes dtype', 3, args[3].int()),
        ('too many points', 6, torch.ones(1, 8, 2, 2, 40, device='cuda') / 80),
    ]
    for what, i, value in bad:
        a = list(args)
        a[i] = value
        if what == 'too many points':
            a[5] = torch.rand(1, 8, 2, 2, 40, 2, device='cuda')
        for mode in ('fused', 'project'):
            if mode == 'project' and what in ('weight cols', 'dtype', 'bias size'):
                continue  # F.linear itself rejects or casts these before our kernel
            try:
                with torch.inference_mode():
                    ma_deform_attn(*a, mode=mode)
            except RuntimeError:
                continue
            raise AssertionError(f'{what} [{mode}]: bad input was accepted')
    grad = [t.clone().requires_grad_(True) if i == 0 else t for i, t in enumerate(args)]
    try:
        ma_deform_attn(*grad)
    except RuntimeError:
        pass
    else:
        raise AssertionError('input that requires grad was accepted')
    try:
        ma_deform_attn(*args, mode='nope')
    except ValueError:
        pass
    else:
        raise AssertionError('unknown mode was accepted')


def test_auto_picks_fused_for_few_queries():
    few = make(1, 50, 8, 32, 256, [(80, 80), (40, 40), (20, 20)], 4)
    many = make(8, 900, 8, 32, 256, [(80, 80), (40, 40), (20, 20)], 4)
    assert choose_mode(few[0], few[1], few[5]) == 'fused'
    assert choose_mode(many[0], many[1], many[5]) == 'project'


def test_auto_limits_follow_the_gpu_generation():
    get = torch.cuda.get_device_capability
    try:
        for cap, table in (((8, 9), marearts_tinydeform.FUSED_LIMIT), ((7, 5), marearts_tinydeform.FUSED_LIMIT_SM75)):
            torch.cuda.get_device_capability = lambda device=None, cap=cap: cap
            assert marearts_tinydeform._limits.__wrapped__(0) is table
    finally:
        torch.cuda.get_device_capability = get


if __name__ == '__main__':
    print('marearts-tinydeform', marearts_tinydeform.__version__, 'CUDA' if CUDA else 'CPU only', torch.__version__)
    for fn in (test_cpu_path_is_reference, test_cuda_float32, test_cuda_half_and_bfloat16,
               test_unaligned_memory_uses_scalar_kernel, test_far_outside_and_empty, test_rejects_bad_input,
               test_auto_picks_fused_for_few_queries, test_auto_limits_follow_the_gpu_generation):
        fn()
        print('ok', fn.__name__)
