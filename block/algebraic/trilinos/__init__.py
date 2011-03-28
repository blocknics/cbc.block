def _init():
    import sys
    if 'dolfin' in sys.modules and not 'PyTrilinos' in sys.modules:
        raise RuntimeError('must be imported before dolfin -- add "import PyTrilinos" first in your script')
    del sys

    import block.algebraic
    class active_backend(object):
        name = 'trilinos'
        def __call__(self):
            import sys
            return sys.modules[self.__module__]
    if block.algebraic.active_backend and block.algebraic.active_backend.name != 'trilinos':
        raise ImportError, 'another backend is already active'
    block.algebraic.active_backend = active_backend()

    # To be able to use ML we must instruct Dolfin to use the Epetra backend.
    import dolfin
    dolfin.parameters["linear_algebra_backend"] = "Epetra"
_init()

from MLPrec import ML
from AztecOO import AztecSolver
from IFPACK import *
from Epetra import *
