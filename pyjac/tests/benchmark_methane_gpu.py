"""Benchmark the generated scalar and cooperative CUDA Jacobians on methane ignition.

Run from the checkout with Python 3, NumPy, Cantera 3 and nvcc on PATH:
    python3 pyjac/tests/benchmark_methane_gpu.py
Use --prepare-only to exercise mechanism generation and reactor sampling on a CPU.
"""

import argparse
import errno
import hashlib
import os
import re
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from time import monotonic
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


def build_step(directory, command, label):
    print(label, flush=True)
    start = monotonic()
    with subprocess.Popen(command, cwd=directory) as process:
        while True:
            try:
                status = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print(f'{label} (still running, {monotonic() - start:.0f} s)', flush=True)
    if status:
        raise subprocess.CalledProcessError(status, command)
    print(f'  finished in {monotonic() - start:.1f} s', flush=True)


def build(directory, destination, arch):
    sources = ['jacob.cu', 'chem_utils.cu', 'spec_rates.cu', 'rxn_rates.cu',
               'rxn_rates_pres_mod.cu', 'gpu_memory.cu']
    for subdir, manifest in [('jacobs', 'jac_list_cuda'), ('rates', 'rate_list_cuda')]:
        names = (directory / subdir / manifest).read_text().split()
        sources.extend(str(Path(subdir) / name) for name in names)
    driver = Path(__file__).with_suffix('.cu').resolve()
    sources = [directory / source for source in sources] + [driver]
    flags = ['nvcc', '-O3', '-std=c++14', '-arch=' + arch]
    objects = []
    for i, source in enumerate(sources, start=1):
        obj = f'unit_{i}.o'
        build_step(destination, flags + ['-dc', '-I', str(directory), str(source), '-o', obj],
                   f'[{i}/{len(sources)}] {source.name}')
        objects.append(obj)
    build_step(destination, flags + ['-rdc=true', *objects, '-o', 'benchmark'],
               'Linking benchmark')
    for obj in objects:
        (destination / obj).unlink()


def source_key(root):
    import cantera as ct
    import numpy as np

    mechanism = next(Path(d) / 'gri30.yaml' for d in ct.get_data_directories()
                     if (Path(d) / 'gri30.yaml').is_file())
    digest = hashlib.sha256(f'{ct.__version__}/{np.__version__}'.encode())
    files = [mechanism, Path(__file__), root / 'pyjac/utils.py']
    for path in files + sorted((root / 'pyjac/core').glob('*.py')):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:20]


def binary_key(arch):
    driver = Path(__file__).with_suffix('.cu').resolve()
    digest = hashlib.sha256(driver.read_bytes())
    digest.update(arch.encode())
    digest.update(subprocess.check_output(['nvcc', '--version']))
    if arch == 'native':
        # nvcc's dry run resolves native to the architectures of visible GPUs.
        plan = subprocess.check_output(['nvcc', '--dryrun', '-arch=native', '-dc', str(driver)],
                                       stderr=subprocess.STDOUT)
        targets = sorted(set(re.findall(rb'(?:sm|compute)_[0-9]+[a-z]?', plan)))
        if targets:
            digest.update(b' '.join(targets))
        else:
            # Some nvcc releases leave "native" unexpanded in dry-run output.
            digest.update(socket.gethostname().encode())
            digest.update(os.environ.get('CUDA_VISIBLE_DEVICES', '').encode())
            if shutil.which('nvidia-smi'):
                devices = subprocess.run(['nvidia-smi', '-L'], capture_output=True)
                digest.update(devices.stdout)
    return digest.hexdigest()[:20]


def cached_directory(path, create):
    if path.is_dir():
        print(f'Reusing {path}', flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a completed directory, including after concurrent first runs.
    with TemporaryDirectory(dir=path.parent) as tmp:
        staging = Path(tmp) / 'build'
        staging.mkdir()
        create(staging)
        try:
            staging.rename(path)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not path.is_dir():
                raise
    print(f'Cached {path}', flush=True)


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
    directory = root / 'out' / 'methane_gpu' / source_key(root)
    cached_directory(directory, prepare)
    if args.prepare_only:
        return
    binary = directory / binary_key(args.arch)
    cached_directory(binary, lambda dest: build(directory, dest, args.arch))
    for systems in args.systems:
        subprocess.run([str(binary / 'benchmark'), str(systems), str(args.repeats),
                        str(args.iterations), *map(str, args.threads)], cwd=directory, check=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Benchmark interrupted.', file=sys.stderr)
        sys.exit(130)
