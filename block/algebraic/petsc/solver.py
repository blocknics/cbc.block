from block.block_base import block_base
from petsc4py import PETSc

class LU(block_base):
    def __init__(self, A):
        self.A = A

        ksp = PETSc.KSP().create()
        ksp.setOperators(A.down_cast().mat())
        ksp.setType('preonly')
        #ksp.setConvergenceHistory()
        ksp.getPC().setType('lu')
        self.ksp = ksp

    def matvec(self, b):
        from dolfin import GenericVector
        if not isinstance(b, GenericVector):
            return NotImplemented()
        if self.A.size(0) != len(b):
            raise RuntimeError(
                'incompatible dimensions for Petsc matvec, %d != %d'%(len(self.b),len(b)))
        x = self.A.create_vec(dim=1)

        self.ksp.solve(b.down_cast().vec(), x.down_cast().vec())

        return x
