from dolfin import MPI
from urllib.parse import quote_plus
import atexit
import inspect
import numpy as np
import os
import pathlib
import pickle

class RegressionFailure(Exception):
    pass

_errs = 0
def _log_or_raise(msg):
    if os.environ.get('BLOCK_REGRESSION_ABORT'):
        raise RegressionFailure(msg)
    else:
        print('!', msg)
        global _errs
        _errs += 1
        if _errs == 1:
            @atexit.register
            def print_msg():
                print(f'! {_errs} expected test(s) failed')
                print('  Run with BLOCK_REGRESSION_SAVE=1 to store new values as reference')

def check_expected(name, vec, show=False, rtol=1e-10, prefix=None):
    if prefix is None:
        prefix = pathlib.Path(inspect.stack()[1].filename).stem
    cur_norm = vec.norm('l2')
    if show:
        print(f'Norm of {name}: {cur_norm:.6f}')
    cur_vec = vec.get_local()
    if not isinstance(cur_vec, np.ndarray):
        cur_vec = np.concatenate(cur_vec)
    # To save disk&repo space, we decimate the vector and calculate mean+rms
    # within each chunk. This is quite arbitrary, but is intended to be robust
    # decimation which still catches most regression scenarios in practice.
    chunks = np.arange(0, len(cur_vec), 10)
    divisor = np.add.reduceat(np.ones_like(cur_vec), chunks)
    cur_vec_mean = np.add.reduceat(cur_vec, chunks) / divisor
    cur_vec_rms = np.sqrt(np.add.reduceat(cur_vec**2, chunks) / divisor)
    cur_vec = (cur_vec_mean + cur_vec_rms) / 2

    fname = pathlib.Path(__file__).parent.parent / f'data/regression/{quote_plus(prefix)}.{quote_plus(name)}.pickle'
    do_check = not os.environ.get('BLOCK_REGRESSION_SAVE') and fname.exists()
    is_serial = MPI.size(MPI.comm_world) == 1
    if do_check:
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        ref_norm = data['norm']
        if is_serial:
            try:
                ref_vec = data['vec']
                err_norm = np.sqrt(np.sum((ref_vec - cur_vec) ** 2) / len(ref_vec))
            except Exception:
                _log_or_raise(f'Could not compute error norm for {name} ({prefix})')
            else:
                rdiff = err_norm / max(ref_norm,1)
                if rdiff > rtol:
                    _log_or_raise(f'Norm of {name} (decimated) error: {err_norm:.4g} ({rdiff:.3g} > {rtol:.3g}) ({prefix})')
        rdiff = abs(cur_norm - ref_norm) / max(ref_norm,1)
        if rdiff > rtol:
            _log_or_raise(f'Norm of {name} {cur_norm:.6g} != {ref_norm:.6g} ({rdiff:.3g} > {rtol:.3g})')
    else:
        if is_serial:
            with open(fname, 'wb') as f:
                data = dict(norm=cur_norm)
                data['vec'] = cur_vec
                pickle.dump(data, f)
        else:
            print('Not saving regression results in parallel')

    # return vec so that it can be used in an expression
    return vec
