"""Exercise generated column ownership and barriers using an OpenMP host block."""

import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from ..core.create_jacobian import create_jacobian


def test_block_jacobian(tmp_path):
    if not all(shutil.which(tool) for tool in ('gcc', 'g++')):
        pytest.skip('host block test uses GCC and OpenMP')

    mechanism = str(Path(__file__).resolve().parents[2] / 'data/h2o2.inp')
    cpu, gpu = tmp_path / 'cpu', tmp_path / 'gpu'
    # A reactive dependent species exercises the extra energy-equation terms.
    for lang, path in [('c', cpu), ('cuda', gpu)]:
        create_jacobian(lang, mechanism, build_path=str(path), last_spec='H2O',
                        cuda_block=(lang == 'cuda'))
    sources = ['jacob', 'dydt', 'chem_utils', 'rxn_rates', 'rxn_rates_pres_mod', 'spec_rates']
    subprocess.check_call(['gcc', '-shared', '-fPIC', '-o', 'ref.so'] +
                          [str(cpu / (name + '.c')) for name in sources] + ['-lm'], cwd=tmp_path)

    # Compile the actual generated CUDA translation unit as C++. Map block
    # barriers to host barriers and discard unused scalar CUDA functions.
    (tmp_path / 'cuda.h').write_text('''
#pragma once
#include <omp.h>
struct dim { unsigned x; };
#define threadIdx (dim{(unsigned)omp_get_thread_num()})
#define blockDim (dim{(unsigned)omp_get_num_threads()})
#define blockIdx (dim{0})
#define gridDim (dim{1})
#define __device__
#define __shared__
#define __syncthreads() _Pragma("omp barrier")
#define GPU_MACROS_CUH
#define GRID_DIM blockDim.x
#define INDEX(i) (threadIdx.x + (i) * GRID_DIM)
''')
    (tmp_path / 'cuda_runtime.h').touch()
    (tmp_path / 'block.cpp').write_text('''
#include "jacob.cu"
extern "C" __attribute__((visibility("default"))) void evaluate(int threads, double p, const double* y, double* jac) {
    jacob_block_workspace work;
    #pragma omp parallel num_threads(threads)
    eval_jacob_block(p, y, jac, work);
}
''')
    subprocess.check_call(['g++', '-std=c++11', '-shared', '-fPIC', '-fopenmp',
                           '-ffunction-sections', '-fvisibility=hidden',
                           '-Wl,--gc-sections', '-I.', '-I' + str(gpu),
                           'block.cpp', '-o', 'block.so'], cwd=tmp_path)

    ref = ctypes.CDLL(str(tmp_path / 'ref.so'))
    block = ctypes.CDLL(str(tmp_path / 'block.so')).evaluate
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    block.argtypes = [ctypes.c_int, ctypes.c_double, array, array]
    ref.eval_jacob.argtypes = ref.dydt.argtypes = [ctypes.c_double, ctypes.c_double, array, array]
    ref.eval_jacob.restype = ref.dydt.restype = block.restype = None
    for temperature in [850., 1500.]:
        y = np.arange(1, 10) * 0.01
        y[0] = temperature
        p = 2e5
        expected = np.zeros(y.size ** 2)
        ref.eval_jacob(0, p, y, expected)
        direction = y * 0.1
        step = 1e-5
        plus, minus = np.empty_like(y), np.empty_like(y)
        ref.dydt(0, p, y + step * direction, plus)
        ref.dydt(0, p, y - step * direction, minus)
        for threads in [1, 3, 9, 32]:
            actual = np.full_like(expected, np.nan)
            block(threads, p, y, actual)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            product = actual.reshape((y.size, y.size), order='F').dot(direction)
            np.testing.assert_allclose(product, (plus - minus) / (2 * step),
                                       rtol=1e-5, atol=1e-5)
