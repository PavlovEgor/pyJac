"""Benchmark the generated scalar and cooperative CUDA Jacobians on methane ignition.

Run from the checkout with Python 3, NumPy, Cantera 3 and nvcc on PATH:
    python3 pyjac/tests/benchmark_methane_gpu.py
Use --prepare-only to exercise mechanism generation and reactor sampling on a CPU.
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('expected a positive integer')
    return number


def prepare(directory):
    import cantera as ct
    import numpy as np
    from pyjac.core.create_jacobian import create_jacobian
    from pyjac.utils import get_species_mappings

    gas = ct.Solution('gri30.yaml')
    gas.TP = 1500., ct.one_atm
    gas.set_equivalence_ratio(1., 'CH4', 'O2:1,N2:3.76')
    reactor = ct.IdealGasConstPressureReactor(gas)
    network = ct.ReactorNet([reactor])
    order, _ = get_species_mappings(gas.n_species, gas.species_index('N2'))
    states = []
    for time in np.linspace(0., 0.003, 32):
        network.advance(time)
        states.append(np.r_[reactor.T, reactor.Y[order[:-1]]])
    np.asarray(states, dtype=np.float64).tofile(directory / 'states.bin')
    print(f'GRI-Mech 3.0, CH4/air, phi=1, 1 atm, 32 ignition snapshots, '
          f'0–3 ms, {states[0][0]:.0f}–{states[-1][0]:.0f} K', flush=True)
    create_jacobian('cuda', mech_name='gri30.yaml', last_spec='N2',
                    cuda_block=True, build_path=str(directory))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--systems', type=positive, nargs='+', default=[1, 64, 1024])
    parser.add_argument('--threads', type=positive, nargs='+', default=[32, 64, 128],
                        help='threads per cooperative block (scalar uses generated default)')
    parser.add_argument('--repeats', type=positive, default=5, help='timing samples per kernel')
    parser.add_argument('--iterations', type=positive, default=10, help='launches per sample')
    parser.add_argument('--arch', default='native', help='nvcc architecture, e.g. sm_86')
    parser.add_argument('--prepare-only', action='store_true', help='skip compilation and timing')
    args = parser.parse_args()
    if not args.prepare_only and not shutil.which('nvcc'):
        parser.error('nvcc is required for GPU timing')
    # The generated scalar code uses 32-bit indices for its strided dense matrix.
    if max(args.systems) > (2**31 // (53 * 53)) - 64 or max(args.threads) > 1024:
        parser.error('batch or thread count exceeds CUDA kernel indexing limits')
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    with TemporaryDirectory(prefix='pyjac-methane-') as tmp:
        directory = Path(tmp)
        prepare(directory)
        if args.prepare_only:
            return
        sources = ['jacob.cu', 'chem_utils.cu', 'spec_rates.cu', 'rxn_rates.cu',
                   'rxn_rates_pres_mod.cu', 'gpu_memory.cu']
        for subdir, manifest in [('jacobs', 'jac_list_cuda'), ('rates', 'rate_list_cuda')]:
            names = (directory / subdir / manifest).read_text().split()
            sources.extend(str(Path(subdir) / name) for name in names)
        driver = Path(__file__).with_suffix('.cu').resolve()
        print('Compiling the generated Jacobians with nvcc ...', flush=True)
        subprocess.run(['nvcc', '-O3', '-std=c++14', '-rdc=true', '-arch=' + args.arch,
                        '-I.', *sources, str(driver), '-o', 'benchmark'], cwd=tmp, check=True)
        for systems in args.systems:
            subprocess.run([str(directory / 'benchmark'), str(systems), str(args.repeats),
                            str(args.iterations), *map(str, args.threads)], cwd=tmp, check=True)


if __name__ == '__main__':
    main()
