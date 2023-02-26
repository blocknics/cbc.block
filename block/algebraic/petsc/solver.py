from block import block_mat, block_vec
from block.block_base import block_base, block_container
from block.object_pool import vec_pool
from dolfin import GenericVector, Matrix, PETScVector
import numpy as np
from petsc4py import PETSc
from .precond import petsc_base, mat as single_mat, vec as single_vec

# VecNest is a perfect fit for block_vec, but I can't get it to work (gives
# error code 86), and I don't know how to get proper error messages or traces
# from PETSc. When set to False, it works but
# - is very inefficient (lots of copying of vectors)
# - doesn't work in parallel
USE_VECNEST=False

def mat(A, _sizes=None):
    if isinstance(A, block_mat):
        # We need sizes upfront in case there are matrix-free blocks, creating
        # a vector to find the size is wasteful but easy
        row_sz = [(v.local_size(), v.size()) for v in A.create_vec(dim=1)]
        col_sz = [(v.local_size(), v.size()) for v in A.create_vec(dim=0)]
        m = PETSc.Mat().createNest([[mat(m, (row_sz[r],col_sz[c])) for (c,m) in enumerate(row)]
                                    for (r,row) in enumerate(A.blocks)])
        if USE_VECNEST:
            m.setVecType(PETSc.Vec.Type.NEST)
        return m
    elif isinstance(A, block_base) or np.isscalar(A):
        if not _sizes:
            raise ValueError('Cannot create unsized PETSc matrix')
        Ad = PETSc.Mat().createPython(_sizes)
        Ad.setPythonContext(petsc_py_wrapper(A))
        Ad.setUp()
        return Ad
    elif isinstance(A, Matrix):
        return single_mat(A)
    else:
        raise TypeError(str(type(A)))

def vec(v):
    if isinstance(v, block_vec):
        if USE_VECNEST:
            return PETSc.Vec().createNest([single_vec(x) for x in v])
        else:
            vecs = [vec(x) for x in v]
            sizes = [x.getSizes() for x in vecs]
            vs = PETSc.Vec().create(comm=vecs[0].comm)
            vs.setType(PETSc.Vec.Type.STANDARD)
            vs.setSizes(sum(s[1] for s in sizes))
            arr = vs.getArray()
            i0 = 0
            for vv in vecs:
                arr2 = vv.getArray()
                arr[i0:i0+len(arr2)] = arr2
                i0 += len(arr2)
            return vs
    elif isinstance(v, GenericVector):
        return single_vec(v)
    else:
        raise TypeError(str(type(v)))

def Vec(v, creator):
    if isinstance(v, PETSc.Vec):
        if USE_VECNEST:
            if v.type == PETSc.Vec.Type.NEST:
                return block_vec([Vec(x) for x in v])
            else:
                return PETScVector(v)
        else:
            if isinstance(creator, block_container):
                ret = creator.create_vec(dim=1)
                arr = v.getArray()
                i0 = 0
                for vv in ret:
                    vv.set_local(arr[i0:i0+len(vv)])
                    i0 += len(vv)
                return ret
            else:
                return PETScVector(v)
    else:
        raise TypeError(str(type(v)))

class petsc_py_wrapper:
    # Python context for cbc.block actions (PC.Type.PYTHON, Mat.Type.PYTHON)
    def __init__(self, A):
        self.A = A

    def mult(self, mat, x, y):
        new_y = vec(self.A * Vec(x, self.A))
        y.aypx(0, new_y)

    def apply(self, pc, x, y):
        return self.mult(None, x, y)

    def setUp(self, pc):
        pass

class petsc_solver(petsc_base):
    def __init__(self, A, precond, V, ksp_type, prefix, options, defaults={}):
        self.A = A
        self.Ad = mat(A)

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
            pc.setPythonContext(petsc_py_wrapper(precond))

    def __str__(self):
        return '<%s ksp of %s>'%(self.__class__.__name__, str(self.A))

    def matvec(self, b):
        self.petsc_op.setUp()
        x = self.Ad.createVecLeft()
        self.petsc_op.solve(vec(b), x)
        return Vec(x, self.A)

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
