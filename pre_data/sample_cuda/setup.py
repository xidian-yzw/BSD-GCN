import sys
import torch.cuda
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME

CXX_FLAGS = ['/sdl', '/permissive-'] if sys.platform == 'win32' else ['-g', '-Werror']


if torch.cuda.is_available() and CUDA_HOME is not None:
    ext_modules = [
        CUDAExtension(
            name ='GCN_Sample',
            module = 'GCN_Sample',
            sources=['src/All/sample_cuda.cpp','src/All/sample_cuda_kernel.cu'],
            extra_compile_args={
                'cxx': [],
                'nvcc': ['-O2']}),
        CUDAExtension(
            name ='GCN_Sample_split',
            module = 'GCN_Sample_split',
            sources=['src/Split/sample_split_cuda.cpp','src/Split/sample_split_cuda_kernel.cu'],
            extra_compile_args={
                'cxx': [],
                'nvcc': ['-O2']}),
    ]


setup(
    name='GCNTool',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})