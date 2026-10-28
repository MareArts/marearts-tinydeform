# Contributing

Thanks for helping. Issues and pull requests are welcome.

- **Tests:** `python -m pytest tests` (or `python tests/test_correctness.py`) must pass on a CUDA GPU. New behaviour comes with a test against `ma_deform_attn_reference`.
- **Speed claims:** include the `bench/bench.py` table before and after, on the same GPU, with the GPU name, driver and PyTorch version.
- **Sign-off:** every commit carries a Developer Certificate of Origin sign-off (`git commit -s`), stating that you wrote the change or have the right to submit it under the project licence (<https://developercertificate.org>).
- **Licence:** contributions are accepted under the Apache License 2.0, the licence of this repository.
