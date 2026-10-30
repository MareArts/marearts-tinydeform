# RTX 2080 Super (Turing, Windows)

`python bench/bench.py --queries 50 100 200 300 600 900 --batch 1 4 8 --dtype float32 float16 bfloat16`, RTX 2080 Super with Max-Q Design (laptop, 8 GB, compute capability 7.5), Windows 11, driver 617.14, PyTorch 2.14.1 + CUDA 12.6, extension built with MSVC 14.44 and CUDA 12.6 for sm_75, marearts-tinydeform 0.1.1. The Deformable-DETR CUDA op was not installed on this machine, so the "official" columns are empty. Times in µs, best of 7 replays of a CUDA graph of 100 calls. bfloat16 differences are of the same size as on the RTX 4090: the PyTorch path itself runs in bfloat16.

In all 54 cases the mode `auto` picked was within 10 % of the faster of the two. With the RTX 4090 thresholds of 0.1.0 it was within 10 % in 133 of 168 cases of a denser sweep (50–900 queries in 14 steps, batch 1 / 2 / 4 / 8); worst, bfloat16 batch 4–8 with 175 queries, 3.7x slower than `fused`.

NVIDIA GeForce RTX 2080 Super with Max-Q Design, torch 2.14.1+cu126 CUDA 12.6, tokens 8400, C 256, heads 8, levels 3, points 4, TF32 off
| dtype | batch | queries | pytorch | official | fused | project | auto (picks) | auto vs pytorch | auto vs official | max abs diff vs pytorch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| float32 | 1 | 50 | 323.1 | — | 45.4 | 189.4 | 45.4 (fused) | 7.1x | — | 1.8e-06 |
| float32 | 1 | 100 | 368.8 | — | 76.1 | 263.9 | 76.1 (fused) | 4.8x | — | 2.2e-06 |
| float32 | 1 | 200 | 413.9 | — | 133.3 | 222.6 | 133.3 (fused) | 3.1x | — | 3.1e-06 |
| float32 | 1 | 300 | 532.4 | — | 193.5 | 271.6 | 193.5 (fused) | 2.8x | — | 3.2e-06 |
| float32 | 1 | 600 | 804.0 | — | 391.7 | 297.3 | 297.3 (project) | 2.7x | — | 3.2e-06 |
| float32 | 1 | 900 | 1067.3 | — | 572.9 | 321.7 | 321.7 (project) | 3.3x | — | 3.2e-06 |
| float32 | 4 | 50 | 1525.8 | — | 148.1 | 1006.8 | 148.1 (fused) | 10.3x | — | 2.4e-06 |
| float32 | 4 | 100 | 1676.7 | — | 275.4 | 1028.1 | 275.4 (fused) | 6.1x | — | 3.2e-06 |
| float32 | 4 | 200 | 1868.5 | — | 526.6 | 1047.4 | 526.6 (fused) | 3.5x | — | 3.7e-06 |
| float32 | 4 | 300 | 2132.4 | — | 763.8 | 1092.8 | 763.8 (fused) | 2.8x | — | 3.7e-06 |
| float32 | 4 | 600 | 2731.2 | — | 1536.1 | 1169.4 | 1169.4 (project) | 2.3x | — | 4.6e-06 |
| float32 | 4 | 900 | 3329.7 | — | 2304.5 | 1233.0 | 1233.0 (project) | 2.7x | — | 8.0e-06 |
| float32 | 8 | 50 | 2960.6 | — | 288.4 | 1965.2 | 288.4 (fused) | 10.3x | — | 2.6e-06 |
| float32 | 8 | 100 | 3222.1 | — | 542.9 | 2025.6 | 542.9 (fused) | 5.9x | — | 3.3e-06 |
| float32 | 8 | 200 | 3742.3 | — | 1058.4 | 2118.1 | 1058.4 (fused) | 3.5x | — | 3.5e-06 |
| float32 | 8 | 300 | 4176.0 | — | 1563.5 | 2109.0 | 1563.5 (fused) | 2.7x | — | 4.4e-06 |
| float32 | 8 | 600 | 5199.5 | — | 3154.8 | 2305.1 | 2305.1 (project) | 2.3x | — | 4.3e-06 |
| float32 | 8 | 900 | 6450.3 | — | 4725.3 | 2515.1 | 2515.1 (project) | 2.6x | — | 4.7e-06 |
| float16 | 1 | 50 | 139.3 | — | 28.3 | 55.0 | 28.3 (fused) | 4.9x | — | 1.7e-02 |
| float16 | 1 | 100 | 153.6 | — | 47.0 | 59.5 | 47.0 (fused) | 3.3x | — | 2.5e-02 |
| float16 | 1 | 200 | 214.7 | — | 84.0 | 66.1 | 66.1 (project) | 3.2x | — | 2.5e-02 |
| float16 | 1 | 300 | 251.5 | — | 128.7 | 70.0 | 70.0 (project) | 3.6x | — | 2.5e-02 |
| float16 | 1 | 600 | 498.5 | — | 274.6 | 83.6 | 83.6 (project) | 6.0x | — | 2.5e-02 |
| float16 | 1 | 900 | 660.1 | — | 427.7 | 99.1 | 99.1 (project) | 6.7x | — | 2.5e-02 |
| float16 | 4 | 50 | 687.8 | — | 109.8 | 206.8 | 109.8 (fused) | 6.3x | — | 2.1e-02 |
| float16 | 4 | 100 | 758.5 | — | 193.9 | 215.7 | 193.9 (fused) | 3.9x | — | 2.0e-02 |
| float16 | 4 | 200 | 927.4 | — | 390.3 | 239.2 | 239.2 (project) | 3.9x | — | 2.3e-02 |
| float16 | 4 | 300 | 1076.5 | — | 582.0 | 261.2 | 261.2 (project) | 4.1x | — | 2.5e-02 |
| float16 | 4 | 600 | 1557.5 | — | 1147.4 | 344.4 | 344.4 (project) | 4.5x | — | 3.1e-02 |
| float16 | 4 | 900 | 2030.7 | — | 1718.1 | 408.8 | 408.8 (project) | 5.0x | — | 2.6e-02 |
| float16 | 8 | 50 | 1326.5 | — | 204.3 | 453.3 | 204.3 (fused) | 6.5x | — | 1.9e-02 |
| float16 | 8 | 100 | 1509.6 | — | 400.8 | 476.1 | 400.8 (fused) | 3.8x | — | 2.1e-02 |
| float16 | 8 | 200 | 1801.1 | — | 776.0 | 516.1 | 516.1 (project) | 3.5x | — | 2.3e-02 |
| float16 | 8 | 300 | 2121.1 | — | 1154.7 | 563.1 | 563.1 (project) | 3.8x | — | 2.4e-02 |
| float16 | 8 | 600 | 3052.7 | — | 2292.5 | 708.9 | 708.9 (project) | 4.3x | — | 2.8e-02 |
| float16 | 8 | 900 | 4003.2 | — | 3422.5 | 843.8 | 843.8 (project) | 4.7x | — | 3.1e-02 |
| bfloat16 | 1 | 50 | 405.4 | — | 31.6 | 300.7 | 31.6 (fused) | 12.8x | — | 1.1e-01 |
| bfloat16 | 1 | 100 | 422.2 | — | 53.1 | 307.6 | 53.1 (fused) | 7.9x | — | 1.7e-01 |
| bfloat16 | 1 | 200 | 486.7 | — | 92.9 | 325.2 | 92.9 (fused) | 5.2x | — | 1.7e-01 |
| bfloat16 | 1 | 300 | 518.8 | — | 136.4 | 320.0 | 136.4 (fused) | 3.8x | — | 1.7e-01 |
| bfloat16 | 1 | 600 | 741.8 | — | 277.6 | 343.4 | 277.6 (fused) | 2.7x | — | 2.1e-01 |
| bfloat16 | 1 | 900 | 898.8 | — | 422.4 | 353.5 | 353.5 (project) | 2.5x | — | 2.1e-01 |
| bfloat16 | 4 | 50 | 1690.0 | — | 111.2 | 1239.5 | 111.2 (fused) | 15.2x | — | 1.4e-01 |
| bfloat16 | 4 | 100 | 1761.1 | — | 196.6 | 1247.4 | 196.6 (fused) | 9.0x | — | 1.6e-01 |
| bfloat16 | 4 | 200 | 1938.5 | — | 388.9 | 1267.9 | 388.9 (fused) | 5.0x | — | 1.8e-01 |
| bfloat16 | 4 | 300 | 2093.2 | — | 570.6 | 1288.6 | 570.6 (fused) | 3.7x | — | 1.8e-01 |
| bfloat16 | 4 | 600 | 2553.7 | — | 1121.8 | 1351.9 | 1121.8 (fused) | 2.3x | — | 2.0e-01 |
| bfloat16 | 4 | 900 | 3030.8 | — | 1679.2 | 1408.0 | 1408.0 (project) | 2.2x | — | 2.5e-01 |
| bfloat16 | 8 | 50 | 3319.2 | — | 206.4 | 2477.6 | 206.4 (fused) | 16.1x | — | 2.0e-01 |
| bfloat16 | 8 | 100 | 3507.2 | — | 400.0 | 2481.2 | 400.0 (fused) | 8.8x | — | 2.3e-01 |
| bfloat16 | 8 | 200 | 3814.1 | — | 775.6 | 2541.2 | 775.6 (fused) | 4.9x | — | 1.9e-01 |
| bfloat16 | 8 | 300 | 4121.9 | — | 1140.7 | 2591.0 | 1140.7 (fused) | 3.6x | — | 2.1e-01 |
| bfloat16 | 8 | 600 | 5071.1 | — | 2259.4 | 2700.2 | 2259.4 (fused) | 2.2x | — | 2.0e-01 |
| bfloat16 | 8 | 900 | 6003.0 | — | 3373.6 | 2836.9 | 2836.9 (project) | 2.1x | — | 2.5e-01 |
