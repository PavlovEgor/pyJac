"""Emit the contiguous, cooperative CUDA interface for a CONP system."""

import os

from .. import utils
from . import rate_subs


def begin_column(file, column):
    file.write('  if (threadIdx.x == {} % blockDim.x) {{\n'.format(column))


def write_helpers(path, specs, reacs, species_order, reaction_order):
    """Generate device versions of the existing contiguous chemistry helpers."""
    block_path = os.path.join(path, 'jacob_block')
    utils.create_dir(block_path)
    with open(os.path.join(block_path, 'header.h'), 'w') as file:
        file.write('#include "../header.cuh"\n')

    rate_subs.write_chem_utils(block_path, 'c', specs, False, device=True)
    rate_subs.write_rxn_rates(block_path, 'c', specs, reacs, reaction_order, device=True)
    has_pmod = any(r.thd_body or r.pdep for r in reacs)
    if has_pmod:
        rate_subs.write_rxn_pressure_mod(block_path, 'c', specs, reacs,
                                         reaction_order, device=True)
    rate_subs.write_spec_rates(block_path, 'c', specs, reacs, species_order,
                               reaction_order, device=True)

    with open(os.path.join(path, 'jacob_block.cuh'), 'w') as file:
        file.write('''#pragma once
#include "header.cuh"

#ifdef CONP
struct jacob_block_workspace {
    double conc[NSP], spec_rates[NSP], cp[NSP], h[NSP];
    double fwd_rates[FWD_RATES];
#if REV_RATES
    double rev_rates[REV_RATES];
#endif
#if PRES_MOD_RATES
    double pres_mod[PRES_MOD_RATES];
#endif
    double y_N, mw_avg, rho, cp_avg;
};

// Collective call by every thread of a one-dimensional block.
// All pointers and the workspace refer to the same system throughout the block.
__device__ void eval_jacob_block(double pres, const double* y, double* jac,
                                 jacob_block_workspace& work);
#endif
''')

    # Place the entry point in the existing Jacobian translation unit so the
    # normal CUDA library build exports both device functions.
    with open(os.path.join(path, 'jacob.cu'), 'a') as file:
        file.write('''
#include "jacob_block.cuh"
#ifdef CONP
#include "jacob_block/chunks.cuh"
namespace pyjac_block {
#include "jacob_block/chem_utils.c"
#include "jacob_block/rxn_rates.c"
#if PRES_MOD_RATES
#include "jacob_block/rxn_rates_pres_mod.c"
#endif
#include "jacob_block/spec_rates.c"
#include "jacob_block/columns.inc"
}

__device__ void eval_jacob_block(double pres, const double* y, double* jac,
                                 jacob_block_workspace& work) {
    pyjac_block::assemble(pres, y, jac, work);
}
#endif
''')
    return block_path


def write_chunk_list(block_path, count):
    """Expose separate device functions to the assembler and the library build."""
    names = ['block_jacob_{}'.format(i) for i in range(count)]
    with open(os.path.join(block_path, 'chunks.cuh'), 'w') as header:
        for name in names:
            header.write('#include "../jacobs/{}.cuh"\n'.format(name))
    if names:
        manifest = os.path.join(os.path.dirname(block_path), 'jacobs', 'jac_list_cuda')
        with open(manifest, 'a') as file:
            if file.tell():
                file.write(' ')
            file.write(' '.join(name + '.cu' for name in names))


def write_state(file, num_rev, num_pdep):
    file.write('''__device__ void assemble(double pres, const double* y, double* jac,
                         jacob_block_workspace& work) {
  const double T = y[0];
  double* conc = work.conc;
  double* fwd_rates = work.fwd_rates;
  double* spec_rates = work.spec_rates;
  double* h = work.h;
  double* cp = work.cp;
''')
    file.write('  double* rev_rates = {};\n'.format('work.rev_rates' if num_rev else '0'))
    file.write('  double* pres_mod = {};\n'.format('work.pres_mod' if num_pdep else '0'))
    file.write('''  for (unsigned i = threadIdx.x; i < NSP * NSP; i += blockDim.x)
    jac[i] = 0.0;
  if (threadIdx.x == 0) {
    eval_conc(T, pres, y + 1, &work.y_N, &work.mw_avg, &work.rho, conc);
    eval_rxn_rates(T, pres, conc, fwd_rates, rev_rates);
''')
    if num_pdep:
        file.write('    get_rxn_pres_mod(T, pres, conc, pres_mod);\n')
    file.write('''    eval_spec_rates(fwd_rates, rev_rates, pres_mod, spec_rates, spec_rates + NSP - 1);
    eval_h(T, h);
    eval_cp(T, cp);
    double cp_avg = 0.0;
    for (int i = 0; i < NSP - 1; ++i)
      cp_avg += y[i + 1] * cp[i];
    work.cp_avg = cp_avg + work.y_N * cp[NSP - 1];
  }
  __syncthreads();
  if (threadIdx.x < NSP) {
  const double mw_avg = work.mw_avg, rho = work.rho, y_N = work.y_N;
  const double cp_avg = work.cp_avg;
''')
