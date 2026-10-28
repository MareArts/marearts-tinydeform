# Copyright 2026 MareArts
# SPDX-License-Identifier: Apache-2.0
# Build: pip install --no-build-isolation .   (needs PyTorch with CUDA and a matching CUDA toolkit)
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    ext_modules=[
        CUDAExtension(
            'marearts_tinydeform._C',
            ['csrc/ma_deform_fused.cu'],
            extra_compile_args={'cxx': ['-O3', '-g0'], 'nvcc': ['-O3']},
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
