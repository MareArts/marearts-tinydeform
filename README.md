# marearts-tinydeform

Value projection + multi-scale deformable attention for DETR-style decoders (RT-DETR, D-FINE, DINO, Deformable-DETR and others), as one CUDA call for inference. Same result as the usual PyTorch code, less work when the number of queries is small.

```python
from marearts_tinydeform import ma_deform_attn

# inside a decoder cross-attention layer, instead of value_proj + the core function
out = ma_deform_attn(memory, self.value_proj.weight, self.value_proj.bias,
                     spatial_shapes, level_start_index, sampling_locations, attention_weights)
out = self.output_proj(out)
```

## Why

A deformable-attention decoder layer projects **every** memory token (`value = value_proj(memory)`, 8,400 tokens for a 640×640 RT-DETR) and then reads only a few of them per query: levels × points × 4 bilinear corners. Bilinear sampling and the attention-weighted sum are linear, so for one query and head

```text
out[d] = Σ_s w_s · (W[d] · memory[t_s] + b[d])  =  W[d] · (Σ_s w_s · memory[t_s])  +  b[d] · Σ_s w_s
```

(`s` runs over the corners inside the map; corners outside count 0, exactly like zero padding). `mode='fused'` computes the right-hand side: it gathers the memory rows first and projects one vector per (query, head), so its cost follows queries × samples instead of tokens.

The fused order reads `C` channels per sample instead of `C / heads`, so with many queries per image the usual order (project first) is cheaper again. `mode='project'` is that order, with our own sampling kernel after `F.linear`. `mode='auto'` (the default) picks one by size.

## Results (RTX 4090)

Value projection + deformable attention of one decoder layer, RT-DETR 640×640 shapes (levels 80×80, 40×40, 20×20; C = 256; 8 heads; 4 points), CUDA graphs so Python launch overhead is not counted, TF32 off. "PyTorch path" is `nn.Linear` + `grid_sample`; "Deformable-DETR CUDA op" is the original CUDA kernel after the same `nn.Linear`.

| dtype | batch | queries | PyTorch path | Deformable-DETR CUDA op | marearts-tinydeform (mode picked) | vs PyTorch | vs CUDA op |
|---|---:|---:|---:|---:|---:|---:|---:|
| float32 | 1 | 100 | 117.2 µs | 68.6 µs | 14.6 µs (fused) | 8.0x | 4.7x |
| float32 | 1 | 300 | 131.1 µs | 69.2 µs | 35.2 µs (fused) | 3.7x | 2.0x |
| float32 | 1 | 900 | 178.2 µs | 75.4 µs | 74.0 µs (project) | 2.4x | 1.0x |
| float32 | 8 | 300 | 738.0 µs | 321.0 µs | 260.7 µs (fused) | 2.8x | 1.2x |
| float16 | 1 | 100 | 68.7 µs | 26.0 µs | 9.8 µs (fused) | 7.0x | 2.7x |
| float16 | 1 | 300 | 75.2 µs | 26.3 µs | 22.4 µs (fused) | 3.4x | 1.2x |
| float16 | 1 | 900 | 99.1 µs | 33.5 µs | 26.6 µs (project) | 3.7x | 1.3x |
| float16 | 8 | 300 | 424.5 µs | 98.5 µs | 84.4 µs (project) | 5.0x | 1.2x |

All 54 measured cases (float32 / float16 / bfloat16, batch 1 / 4 / 8, 50–900 queries) are in [bench/results/rtx4090.md](bench/results/rtx4090.md). In every case the mode `auto` picked was within 10 % of the faster of the two, and no slower than the Deformable-DETR CUDA op. This is one layer, not a whole model: what it saves end to end depends on how much of your model's time the decoder takes.

**Measured on two GPUs.** The `auto` thresholds were fitted on the RTX 4090 (used from compute capability 8.0 on) and on an RTX 2080 Super Max-Q laptop GPU under Windows (Turing, used below 8.0), where the crossover comes much earlier in float16 and much later in bfloat16, which has no tensor cores there. RTX 2080 Super results: [bench/results/rtx2080super.md](bench/results/rtx2080super.md). Other GPUs may cross over elsewhere. Run `python bench/bench.py` on yours and pass `mode='fused'` or `mode='project'` if it picks wrong (results from other GPUs are welcome).

## Install

Needs PyTorch with CUDA and a matching CUDA toolkit (`nvcc`).

```bash
git clone https://github.com/MareArts/marearts-tinydeform
cd marearts-tinydeform
pip install --no-build-isolation .
python tests/test_correctness.py        # or: python -m pytest tests
```

## What it computes, and what it does not

- Inputs: `memory [B, N, C]`, `value_weight [H·D, C]`, `value_bias [H·D]` or `None`, `spatial_shapes [L, 2]` (int64, height and width), `level_start_index [L]` (int64), `sampling_locations [B, Q, H, L, P, 2]` (x, y in [0, 1]), `attention_weights [B, Q, H, L, P]` (already softmaxed). Output `[B, Q, H·D]`.
- The result is that of `ma_deform_attn_reference` (in this package): `nn.Linear`, then `grid_sample` with bilinear interpolation, zero padding and `align_corners=False` — the Deformable-DETR / RT-DETR default.
- float32, float16 and bfloat16; accumulation in float32. CPU tensors run the reference.
- Inference only: there is no backward, and inputs that require grad are rejected.
- Not supported: a value padding mask (`value.masked_fill(mask, 0)` after the projection), levels × points > 64, C > 8192, and RT-DETRv2's `discrete_sample` (a different operator that rounds the locations).

## Prior work

Deformable attention is from [Deformable DETR](https://arxiv.org/abs/2010.04159) (Zhu et al., 2021). Sampling features first and projecting the samples appears in [Vision Transformer with Deformable Attention](https://arxiv.org/abs/2201.00520) (Xia et al., 2022) and in the key-aware deformable attention of [Lite DETR](https://arxiv.org/abs/2303.07335) (Li et al., 2023). marearts-tinydeform applies the same linear reordering to the standard DETR decoder: kernels that give the standard result, measurements against the usual implementations, and an automatic choice between the two orders.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Contributions: [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md). Security reports: [.github/SECURITY.md](.github/SECURITY.md).

## About

marearts-tinydeform is made by [MareArts](https://www.marearts.com), a computer vision company building license plate recognition (ANPR) and other vision software.
