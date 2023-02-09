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

def _regr_root():
    return pathlib.Path(__file__).parent.parent / 'data/regression'

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
                print(f'  Remove file(s) in {_regr_root()}/ and re-run to store new values as reference')

def check_expected(name, vec, show=False, rtol=1e-6, itol=0.1, prefix=None, expected=None):
    itol += 1
    if prefix is None:
        prefix = pathlib.Path(inspect.stack()[1].filename).stem
    def _decimate(v):
        # Pickle not supported for block_vec/dolfin_vec, convert to numpy
        if hasattr(v, 'get_local'):
            v = v.get_local()
        if not isinstance(v, np.ndarray):
            v = np.concatenate(v)
        # To save disk&repo space, we decimate the vector and calculate mean+rms
        # within each chunk. This is quite arbitrary, but is intended to be robust
        # decimation which still catches most regression scenarios in practice.
        chunks = np.arange(0, len(v), 10)
        divisor = np.add.reduceat(np.ones_like(v), chunks)
        v_mean = np.add.reduceat(v, chunks) / divisor
        v_rms = np.sqrt(np.add.reduceat(v**2, chunks) / divisor)
        return (v_mean + v_rms) / 2
    def _l2(v):
        if np.isscalar(v):
            return v
        elif hasattr(v, 'norm'):
            return v.norm('l2')
        else:
            return np.sqrt(np.sum(v**2))

    fname = _regr_root() / f'{quote_plus(prefix)}.{quote_plus(name)}.pickle'
    if fname.exists():
        with open(fname, 'rb') as f:
            data = pickle.load(f)
    else:
        data = {}

    cur_norm = _l2(vec)
    cur_iter = getattr(vec, '_regr_test_niter', None)
    ref_iter = data.get('iter')
    if expected is None:
        cur_vec = _decimate(vec)
        ref_vec = data.get('vec')
        ref_norm = data.get('norm')
    else:
        cur_vec = vec
        ref_vec = expected
        ref_norm = _l2(expected)
    is_serial = (MPI.size(MPI.comm_world) == 1)

    if is_serial and ref_vec is not None:
        try:
            err_norm = _l2(cur_vec - ref_vec)
        except Exception as e:
            _log_or_raise(f'Failed to compute error norm: {e}')
        else:
            rdiff = err_norm / max(ref_norm, len(cur_vec))
            if show:
                print(f'Norm of {name}: {cur_norm:.4f}, error: {err_norm:.4g}')
            if rdiff > rtol:
                _log_or_raise(f'Error in {name}: {err_norm:.4g} ({rdiff:.3g} > rtol {rtol:.3g}) ({prefix})')
    else:
        if show:
            print(f'Norm of {name}: {cur_norm:.4f}')
        if ref_norm is not None:
            rdiff = abs(cur_norm - ref_norm) / max(ref_norm, len(cur_vec))
            if rdiff > rtol:
                _log_or_raise(f'Norm of {name} {cur_norm:.6g} != {ref_norm:.6g} ({rdiff:.3g} > rtol {rtol:.3g})')
    if is_serial and ref_iter is not None:
        if cur_iter is None or not ref_iter/itol <= cur_iter <= ref_iter*itol:
            _log_or_raise(f'Solver for {name} used {cur_iter} != ({ref_iter}/{itol}--{ref_iter}*{itol}) iterations ({prefix})')

    if not fname.exists():
        if is_serial:
            data = {}
            if cur_iter is not None:
                data['iter'] = cur_iter
            if expected is None:
                data['norm'] = cur_norm
                data['vec'] = cur_vec
            if data:
                with open(fname, 'wb') as f:
                    pickle.dump(data, f)
        else:
            if MPI.rank(MPI.comm_world) == 0:
                print('Not saving regression results in parallel')

    return vec
