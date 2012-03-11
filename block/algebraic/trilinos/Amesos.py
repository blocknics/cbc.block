from __future__ import division

from block.block_base import block_base

class AmesosSolver(block_base):
    """Trilinos interface to direct solvers. The available solvers depend on
    your Trilinos installation, but the default (Klu) is usually available.

    Serial solvers: Klu, Lapack, Umfpack, Taucs, Superlu

    Parallelsolvers: Superludist, Mumps, Dscpack, Pardiso, Paraklete
    """
    def __init__(self, A, solver='Klu'):
        from PyTrilinos import Epetra, Amesos
        self.A = A # Keep reference
        self.b = A.create_vec(dim=0)
        self.x = A.create_vec(dim=1)
        problem = Epetra.LinearProblem(A.down_cast().mat(),
                                       self.x.down_cast().vec(),
                                       self.b.down_cast().vec())
        self.solver = Amesos.Factory().Create(solver, problem)
        err = self.solver.SymbolicFactorization()
        if err > 1:
            raise RuntimeError("Amesos " + solver + " symbolic factorization failed")
        err = self.solver.NumericFactorization()
        if err > 1:
            raise RuntimeError("Amesos " + solver + " numeric factorization failed")

    def matvec(self, b):
        from dolfin import GenericVector
        if not isinstance(b, GenericVector):
            return NotImplemented()
        if len(self.b) != len(b):
            raise RuntimeError(
                'incompatible dimensions for Amesos matvec, %d != %d'%(len(self.b),len(b)))
        self.b[:] = b
        self.solver.Solve()
        return self.x

    def __str__(self):
        return '<%s solver for %s>'%(self.__class__.__name__, str(self.A))
