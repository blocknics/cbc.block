from dolfin import MPI
from urllib.parse import quote
import atexit
import inspect
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

def check_expected(name, vec, prefix=None, rtol=1e-10, show=True, norm_only=True):
    if prefix is None:
        prefix = pathlib.Path(inspect.stack()[1].filename).stem
    norm = vec.norm('l2')
    if show:
        print(f'Norm of {name}: {norm:.6f}')

    fname = pathlib.Path(__file__).parent.parent / f'data/regression/{quote(prefix)}.{quote(name)}.pickle'
    do_check = not os.environ.get('BLOCK_REGRESSION_SAVE') and fname.exists()
    is_serial = MPI.size(MPI.comm_world) == 1
    if do_check:
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        reference_norm = data['norm']
        if is_serial and not norm_only:
            reference_vec = vec.copy()
            try:
                reference_vec.set_local(data['vec'])
                err_norm = (reference_vec - vec).norm('l2')
            except Exception:
                _log_or_raise(f'Could not compute error norm for {name} ({prefix})')
            else:
                rdiff = err_norm / max(reference_norm,1)
                if rdiff > rtol:
                    _log_or_raise(f'Norm of {name} error: {err_norm:.4g} ({rdiff:.3g} > {rtol:.3g}) ({prefix})')
        else:
            rdiff = abs(norm - reference_norm) / max(reference_norm,1)
            if rdiff > rtol:
                _log_or_raise(f'Norm of {name} {norm:.6g} != {reference_norm:.6g} ({rdiff:.3g} > {rtol:.3g})')
    else:
        if is_serial:
            with open(fname, 'wb') as f:
                data = dict(norm=norm)
                if not norm_only:
                    data['vec'] = vec.get_local()
                pickle.dump(data, f)
        else:
            print('Not saving regression results in parallel')
