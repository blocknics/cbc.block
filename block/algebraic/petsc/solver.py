from block import block_mat, block_vec
from block.block_base import block_base
from block.object_pool import vec_pool
from dolfin import GenericVector, Matrix, PETScVector
import numpy as np
from petsc4py import PETSc
from .precond import petsc_base, mat as single_mat, vec as single_vec

def mat(A, _sizes=None):
    if isinstance(A, block_mat):
        # We need sizes upfront in case there are matrix-free blocks, creating
        # vectors is wasteful but easy
        row_sz = [(v.local_size(), v.size()) for v in A.create_vec(dim=1)]
        col_sz = [(v.local_size(), v.size()) for v in A.create_vec(dim=0)]
        m = PETSc.Mat().createNest([[mat(m, (row_sz[r],col_sz[c])) for (c,m) in enumerate(row)] for (r,row) in enumerate(A.blocks)])
        return m
    elif isinstance(A, block_base) or np.isscalar(A):
        if not _sizes:
            raise ValueError('Cannot create unsized PETSc matrix')
        Ad = PETSc.Mat().createPython(_sizes)
        Ad.setPythonContext(wrap_mat(A))
        Ad.setUp()
        return Ad
    elif isinstance(A, Matrix):
        return single_mat(A)
    else:
        raise TypeError(str(type(A)))

def vec(v):
    if isinstance(v, block_vec):
        return PETSc.Vec().createNest([vec(x) for x in v])
    elif isinstance(v, GenericVector):
        return single_vec(v)
    else:
        raise TypeError(str(type(v)))

def Vec(v):
    if isinstance(v, PETSc.Vec):
        if v.type == PETSc.Vec.Type.NEST:
            return block_vec([Vec(x) for x in v])
        else:
            return PETScVector(v)
    else:
        raise TypeError(str(type(v)))

class wrap_mat:
    def __init__(self, A):
        self.A = A

    def mult(self, mat, x, y):
        new_y = vec(self.A * Vec(x))
        y.aypx(0, new_y)

class wrap_precond:
    def __init__(self, P):
        self.P = P

    def setUp(self, pc):
        #B, P = pc.getOperators()
        #ctx = B.getPythonContext()
        pass

    def apply(self, pc, x, y):
        # x and y are petsc vectors
        new_y = vec(self.P * Vec(x))
        y.aypx(0, new_y)

class petsc_solver(petsc_base):
    def __init__(self, A, precond, V, ksp_type, prefix, options, defaults={}):
        self.A = A
        self.Ad = mat(A)
        print(f'# petsc matrix type: {self.Ad.getType()}')
        if self.Ad.getType() == 'nest':
            for i in range(self.A.blocks.shape[0]):
                for j in range(self.A.blocks.shape[1]):
                    print(f'#  {i,j} -> {self.Ad.getNestSubMatrix(0,0).getType()}')

        prefix, self.optsDB = self._merge_options(prefix=prefix, options=options, defaults=defaults)

        self.petsc_op = PETSc.KSP().create(V.mesh().mpi_comm() if V else None)
        self.petsc_op.setOptionsPrefix(prefix)
        if ksp_type is not None:
            self.petsc_op.setType(ksp_type)
        self.petsc_op.setOperators(self.Ad)
        self.petsc_op.setFromOptions()
        if precond is not None:
            pc = self.petsc_op.getPC()
            pc.setType(PETSc.PC.Type.PYTHON)
            pc.setPythonContext(wrap_precond(precond))

    def __str__(self):
        return '<%s ksp of %s>'%(self.__class__.__name__, str(self.A))

    def matvec(self, b):
        self.petsc_op.setUp()
        x = self.A.create_vec(dim=1)
        print('# petsc sizes (A, b, x):', self.petsc_op.getOperators()[0].getSizes(), vec(b).getSizes(), vec(x).getSizes())
        self.petsc_op.solve(vec(b), vec(x))
        return x

    @vec_pool
    def create_vec(self, dim=1):
        from dolfin import PETScVector
        if dim == 0:
            m = self.Ad.createVecRight()
        elif dim == 1:
            m = self.Ad.createVecLeft()
        else:
            raise ValueError('dim must be 0 or 1')
        return PETScVector(m)

class KSP(petsc_solver):
    def __init__(self, A, precond=None, prefix=None, **parameters):
        super().__init__(A, precond=precond, V=None, ksp_type=None, prefix=prefix, options=parameters,
                         defaults={
                         })
