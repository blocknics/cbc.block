from __future__ import division

from block.block_base import block_base
from petsc4py import PETSc

class precond(block_base):
    def __init__(self, A, prectype, parameters=None, pdes=1, nullspace=None):
        from dolfin import info
        from time import time

        T = time()
        Ad = A.down_cast().mat()

        if nullspace:
            from block.block_util import isscalar
            ns = PETSc.NullSpace()
            if isscalar(nullspace):
                ns.create(constant=True)
            else:
                ns.create(constant=False, vectors=[v.down_cast().vec() for v in nullspace])
            try:
                Ad.setNearNullSpace(ns)
            except:
                info('failed to set near null space (not supported in petsc4py version)')

        self.A = A
        self.ml_prec = PETSc.PC()
        self.ml_prec.create()
        self.ml_prec.setType(prectype)
#        self.ml_prec.setOperators(Ad, Ad, PETSc.Mat.Structure.SAME_PRECONDITIONER)
        self.ml_prec.setOperators(Ad, Ad) 

        # Merge parameters into the options database
        if parameters:
            origOptions = PETSc.Options().getAll()
            for key,val in parameters.iteritems():
                PETSc.Options().setValue(key, val)

        # Create preconditioner based on the options database
        self.ml_prec.setFromOptions()
        self.ml_prec.setUp()

        # Reset the options database
        if parameters:
            for key in parameters.iterkeys():
                PETSc.Options().delValue(key)
            for key,val in origOptions.iteritems():
                PETSc.Options().setValue(key, val)

        info('constructed %s preconditioner in %.2f s'%(self.__class__.__name__, time()-T))

    def matvec(self, b):
        from dolfin import GenericVector
        if not isinstance(b, GenericVector):
            return NotImplemented
        x = self.A.create_vec(dim=1)
        if len(x) != len(b):
            raise RuntimeError(
                'incompatible dimensions for PETSc matvec, %d != %d'%(len(x),len(b)))

        self.ml_prec.apply(b.down_cast().vec(), x.down_cast().vec())
        return x

    def down_cast(self):
        return self.ml_prec

    def __str__(self):
        return '<%s prec of %s>'%(self.__class__.__name__, str(self.A))

class ML(precond):
    def __init__(self, A, parameters=None, pdes=1, nullspace=None):
        options = {
            # Symmetry- and PD-preserving smoother
            'mg_levels_ksp_type': 'chebyshev',
            'mg_levels_pc_type':  'jacobi',
            # Fixed number of iterations to preserve linearity
            'mg_levels_ksp_max_it':               2,
            'mg_levels_ksp_check_norm_iteration': 9999,
            # Exact inverse on coarse grid
            'mg_coarse_ksp_type': 'preonly',
            'mg_coarse_pc_type':  'lu',
            }
        options.update(PETSc.Options().getAll())
        if parameters:
            options.update(parameters)
        precond.__init__(self, A, PETSc.PC.Type.ML, options, pdes, nullspace)

class ILU(precond):
    def __init__(self, A, parameters=None, pdes=1, nullspace=None):
        precond.__init__(self, A, PETSc.PC.Type.ILU, parameters, pdes, nullspace)

class Cholesky(precond):
    def __init__(self, A, parameters=None):
        precond.__init__(self, A, PETSc.PC.Type.CHOLESKY, parameters, 1, None)

class LU(precond):
    def __init__(self, A, parameters=None):
        precond.__init__(self, A, PETSc.PC.Type.LU, parameters, 1, None)

class MumpsSolver(LU):
    def __init__(self, A, parameters=None):
        options = parameters.copy() if parameters else {}
        options['pc_factor_mat_solver_package'] = 'mumps'
        precond.__init__(self, A, PETSc.PC.Type.LU, parameters, 1, None)
