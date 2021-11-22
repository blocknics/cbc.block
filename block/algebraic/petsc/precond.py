from __future__ import division

from builtins import str
from block.block_base import block_base
from block.object_pool import vec_pool
from petsc4py import PETSc
import dolfin as df
import numpy as np
from time import time
from dolfin import info


class precond(block_base):
    def __init__(self, A, prectype, parameters=None, pdes=1, nullspace=None):

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
        self.petsc_prec = PETSc.PC()
        self.petsc_prec.create()
        self.petsc_prec.setType(prectype)
#        self.petsc_prec.setOperators(Ad, Ad, PETSc.Mat.Structure.SAME_PRECONDITIONER)
        self.petsc_prec.setOperators(Ad, Ad) 

        # Merge parameters into the options database
        if parameters:
            origOptions = PETSc.Options().getAll()
            for key,val in iter(list(parameters.items())):
                PETSc.Options().setValue(key, val)

        self.is_setup = False

        # Reset the options database
        if parameters:
            for key in iter(list(parameters.keys())):
                PETSc.Options().delValue(key)
            for key,val in iter(list(origOptions.items())):
                PETSc.Options().setValue(key, val)

    def matvec(self, b):
        # NOTE: we want to be lazy only setup when the action is needed
        if not self.is_setup:
            T = time()
            # Create preconditioner based on the options database
            self.petsc_prec.setFromOptions()
            self.petsc_prec.setUp()
            info('constructed %s preconditioner in %.2f s'%(self.__class__.__name__, time()-T))

            self.is_setup = True
            
        from dolfin import GenericVector
        if not isinstance(b, GenericVector):
            return NotImplemented
        x = self.A.create_vec(dim=1)
        if len(x) != len(b):
            raise RuntimeError(
                'incompatible dimensions for PETSc matvec, %d != %d'%(len(x),len(b)))

        self.petsc_prec.apply(b.down_cast().vec(), x.down_cast().vec())
        return x

    def down_cast(self):
        return self.petsc_prec

    def __str__(self):
        return '<%s prec of %s>'%(self.__class__.__name__, str(self.A))

    def create_vec(self, dim=1):
        return self.A.create_vec(dim)

    
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
        parameters = {} if parameters is None else parameters
        precond.__init__(self, A, PETSc.PC.Type.LU, parameters, 1, None)


class MUMPS_LU(LU):
    def __init__(self, A, parameters=None):
        options = parameters.copy() if parameters else {}
        options['pc_factor_mat_solver_package'] = 'mumps'
        precond.__init__(self, A, PETSc.PC.Type.LU, parameters, 1, None)


class SUPERLU_LU(LU):
    def __init__(self, A, parameters=None):
        options = parameters.copy() if parameters else {}
        options['pc_factor_mat_solver_package'] = 'superlu'
        precond.__init__(self, A, PETSc.PC.Type.LU, parameters, 1, None)
        self.petsc_prec.setFactorPivot(1E-16)

        
class AMG(precond):
    """
    BoomerAMG preconditioner from the Hypre Library
    """
    def __init__(self, A, parameters=None, pdes=1, nullspace=None):
        
        options = {
            "pc_hypre_type": "boomeramg",
            #"pc_hypre_boomeramg_cycle_type": "V", # (V,W)
            #"pc_hypre_boomeramg_max_levels": 25,
            #"pc_hypre_boomeramg_max_iter": 1,
            #"pc_hypre_boomeramg_tol": 0,     
            #"pc_hypre_boomeramg_truncfactor" : 0,      # Truncation factor for interpolation
            #"pc_hypre_boomeramg_P_max": 0,             # Max elements per row for interpolation
            #"pc_hypre_boomeramg_agg_nl": 0,            # Number of levels of aggressive coarsening
            #"pc_hypre_boomeramg_agg_num_paths": 1,     # Number of paths for aggressive coarsening
            #"pc_hypre_boomeramg_strong_threshold": .25,# Threshold for being strongly connected
            #"pc_hypre_boomeramg_max_row_sum": 0.9,
            #"pc_hypre_boomeramg_grid_sweeps_all": 1,   # Number of sweeps for the up and down grid levels 
            #"pc_hypre_boomeramg_grid_sweeps_down": 1,  
            #"pc_hypre_boomeramg_grid_sweeps_up":1,
            #"pc_hypre_boomeramg_grid_sweeps_coarse": 1,# Number of sweeps for the coarse level (None)
            #"pc_hypre_boomeramg_relax_type_all":  "symmetric-SOR/Jacobi", # (Jacobi, sequential-Gauss-Seidel, seqboundary-Gauss-Seidel, 
                                                                          #  SOR/Jacobi, backward-SOR/Jacobi,  symmetric-SOR/Jacobi,  
                                                                          #  l1scaled-SOR/Jacobi Gaussian-elimination, CG, Chebyshev, 
                                                                          #  FCF-Jacobi, l1scaled-Jacobi)
            #"pc_hypre_boomeramg_relax_type_down": "symmetric-SOR/Jacobi",
            #"pc_hypre_boomeramg_relax_type_up": "symmetric-SOR/Jacobi",
            #"pc_hypre_boomeramg_relax_type_coarse": "Gaussian-elimination",
            #"pc_hypre_boomeramg_relax_weight_all": 1,   # Relaxation weight for all levels (0 = hypre estimates, -k = determined with k CG steps)
            #"pc_hypre_boomeramg_relax_weight_level": (1,1), # Set the relaxation weight for a particular level
            #"pc_hypre_boomeramg_outer_relax_weight_all": 1,
            #"pc_hypre_boomeramg_outer_relax_weight_level": (1,1),
            #"pc_hypre_boomeramg_no_CF": "",               # Do not use CF-relaxation 
            #"pc_hypre_boomeramg_measure_type": "local",   # (local global)
            #"pc_hypre_boomeramg_coarsen_type": "Falgout", # (Ruge-Stueben, modifiedRuge-Stueben, Falgout, PMIS, HMIS)
            #"pc_hypre_boomeramg_interp_type": "classical",# (classical, direct, multipass, multipass-wts, ext+i, ext+i-cc, standard, standard-wts, FF, FF1)
            #"pc_hypre_boomeramg_print_statistics": "",
            #"pc_hypre_boomeramg_print_debug": "",
            #"pc_hypre_boomeramg_nodal_coarsen": "",
            #"pc_hypre_boomeramg_nodal_relaxation": "",
            }
        options.update(PETSc.Options().getAll())
        if parameters:
            options.update(parameters)
        precond.__init__(self, A, PETSc.PC.Type.HYPRE, options, pdes, nullspace)


class SOR(precond):
    def __init__(self, A, parameters=None, pdes=1, nullspace=None):
        options = {
            "pc_sor_omega": 1,      # relaxation factor (0 < omega < 2, 1 is Gauss-Seidel)
            "pc_sor_its": 1,        # number of inner SOR iterations
            "pc_sor_lits": 1,       # number of local inner SOR iterations
            "pc_sor_symmetric": "", # for SSOR
            #"pc_sor_backward": "",
            #"pc_sor_forward": "",  
            #"tmp_pc_sor_local_symmetric": "", # use SSOR separately on each processor
            #"tmp_pc_sor_local_backward": "",  
            #"tmp_pc_sor_local_forward": "",
            }
        options.update(PETSc.Options().getAll())
        if parameters:
            options.update(parameters)
        precond.__init__(self, A, PETSc.PC.Type.SOR, options, pdes, nullspace)


class Elasticity(precond):
    def __init__(self, A, parameters=None, pdes=1, nullspace=None):

        prefix = str(time())
        A.down_cast().mat().setOptionsPrefix(prefix)
        
        options = {
            'pc_mg_cycle_type': 'w',
            'pc_mg_multiplicative_cycles': 2
            }
        options = {'_'.join([prefix, key]): val for key, val in options.items()}
        print(options)
        options.update(PETSc.Options().getAll())
        if parameters:
            options.update(parameters)
        precond.__init__(self, A, PETSc.PC.Type.GAMG, options, pdes, nullspace)

class ASM(precond):
    """
    Additive Scwharz Method.
    Defaults (or should default, not tested) to one block per process.
    """
    def __init__(self, A, parameters=None, pdes=1, nullspace=None):
        options = {
            #"pc_asm_blocks":  1,             # Number of subdomains
            "pc_asm_overlap": 1,             # Number of grid points overlap
            "pc_asm_type":  "RESTRICT",      # (NONE, RESTRICT, INTERPOLATE, BASIC)
            "sup_ksp_type": "preonly",       # KSP solver for the subproblems
            "sub_pc_type": "ilu"             # Preconditioner for the subproblems
            }
        options.update(PETSc.Options().getAll())
        if parameters:
            options.update(parameters)
        precond.__init__(self, A, PETSc.PC.Type.ASM, options, pdes, nullspace)


class Jacobi(precond):
    """
    Actually this is only a diagonal scaling preconditioner; no support for relaxation or multiple iterations.
    """
    def __init__(self, A, parameters=None, pdes=1, nullspace=None):
        options = {
            #"pc_jacobi_rowmax": "",  # Use row maximums for diagonal
            #"pc_jacobi_rowsum": "",  # Use row sums for diagonal
            #"pc_jacobi_abs":, "",    # Use absolute values of diagaonal entries
            }
        options.update(PETSc.Options().getAll())
        if parameters:
            options.update(parameters)
        precond.__init__(self, A, PETSc.PC.Type.JACOBI, options, pdes, nullspace)


def vec(x):
    return df.as_backend_type(x).vec()


def mat(A):
    return df.as_backend_type(A).mat()


class HypreAMS(block_base):
    '''
    AMG auxiliary space preconditioner for H(curl) problems in 2d/3d and
    H(div) problems in 2d. The bilinear form behind the matrix whose 
    approximate inverse is constructed is: Find u in V

       alpha*inner(curl(u), curl(v)) + beta*inner(u, v)*dx for all v in V

    where alpha > 0 and beta >= 0.
    '''
    def __init__(self, A, V, ams_zero_beta_poisson=False):
        mesh = V.mesh()
        if mesh.geometry().dim() == 2:
            assert V.ufl_element().family() in ('Raviart-Thomas', 'Nedelec 1st kind H(curl)')
        else:
            assert V.ufl_element().family() == 'Nedelec 1st kind H(curl)'
        # FIXME: only tested this for lower orders
        assert V.ufl_element().degree() == 1
            
        # AMS setup requires auxiliary operators
        Q = df.FunctionSpace(mesh, 'CG', 1)
        G = df.DiscreteOperators.build_gradient(V, Q)

        pc = PETSc.PC().create(mesh.mpi_comm())
        pc.setType('hypre')
        pc.setHYPREType('ams')

        pc.setHYPREDiscreteGradient(mat(G))

        # Constants
        constants = [vec(df.interpolate(df.Constant(c), V).vector())
                     for c in np.eye(mesh.geometry().dim())]

        pc.setHYPRESetEdgeConstantVectors(*constants)

        # NOTE: signal that we don't have lower order term
        ams_zero_beta_poisson and pc.setHYPRESetBetaPoissonMatrix(None)

        pc.setOperators(mat(A))

        pc.setFromOptions()
        pc.setUp()

        self.pc = pc
        self.A = A   # For creating vec

    def matvec(self, b):
        if not isinstance(b, df.GenericVector):
            return NotImplemented
        x = self.A.create_vec(dim=1)

        if x.size() != b.size():
            raise RuntimeError(
                'incompatible dimensions for PETSc matvec, %d != %d'%(len(x),len(b)))

        self.pc.apply(vec(b), vec(x))

        return x

    @vec_pool
    def create_vec(self, dim=1):
        return self.A.create_vec(dim)


class HypreADS(block_base):
    '''
    AMG auxiliary space preconditioner for H(div) problems in 3d. The bilinear 
    form behind the matrix whose approximate inverse is constructed is: Find u in V

       alpha*inner(div(u), div(v)) + beta*inner(u, v)*dx for all v in V

    where alpha > 0 and beta >= 0.
    '''
    @staticmethod
    def coordinates(mesh):
        '''As P1 functions'''
        V = df.FunctionSpace(mesh, 'CG', 1)
        
        return V.tabulate_dof_coordinates()

    @staticmethod
    def discrete_gradient(mesh):
        '''P1 to Ned1 map'''
        Ned = df.FunctionSpace(mesh, 'Nedelec 1st kind H(curl)', 1)
        P1 = df.FunctionSpace(mesh, 'CG', 1)
    
        G = df.DiscreteOperators.build_gradient(Ned, P1)

        return mat(G)
    
    @staticmethod
    def discrete_curl(mesh):
        '''Ned1 to RT1 map'''
        assert mesh.geometry().dim() == 3
        assert mesh.topology().dim() == 3

        Ned = df.FunctionSpace(mesh, 'Nedelec 1st kind H(curl)', 1)
        RT = df.FunctionSpace(mesh, 'RT', 1)

        RT_f2dof = np.array(RT.dofmap().entity_dofs(mesh, 2))
        Ned_e2dof = np.array(Ned.dofmap().entity_dofs(mesh, 1))

        # Alloc sparsity
        a = df.Constant(0)*df.inner(df.TestFunction(RT), df.TrialFunction(Ned))*df.dx
        C = df.PETScMatrix()
        df.assemble(a, tensor=C)

        RT_f2dof = np.array(RT.dofmap().entity_dofs(mesh, 2))
        Ned_e2dof = np.array(Ned.dofmap().entity_dofs(mesh, 1))

        # Facets in terms of edges
        mesh.init(2, 1)
        f2e = mesh.topology()(2, 1)

        row_cols, row_values = np.zeros(3, dtype='int32'), np.array([-2, 2, -2])
        Cmat = C.mat()

        for facet, row in enumerate(RT_f2dof):
            row_cols[:] = Ned_e2dof[f2e(facet)]
            Cmat.setValues([row], row_cols, row_values, PETSc.InsertMode.INSERT_VALUES)
        Cmat.assemble()

        
        return Cmat

    def __init__(self, A, V):
        assert V.mesh().geometry().dim() == 3
        assert V.ufl_element().family() == 'Raviart-Thomas'
        assert V.ufl_element().degree() == 1

        mesh = V.mesh()
        # FIXME: curl matrix needs to be fixed for parallel        
        assert mesh.mpi_comm().size == 1  
        
        C = HypreADS.discrete_curl(mesh)
        G = HypreADS.discrete_gradient(mesh)
        xyz = HypreADS.coordinates(mesh)

        pc = PETSc.PC().create(mesh.mpi_comm())
        pc.setOperators(mat(A))
                    
        pc.setType('hypre')
        pc.setHYPREType('ads')
        # Attach auxiliary operators
        pc.setHYPREDiscreteGradient(G)
        pc.setHYPREDiscreteCurl(G)
        pc.setCoordinates(xyz)
        pc.setUp()
    
        self.pc = pc
        self.A = A   # For creating vec
 
    def matvec(self, b):
        if not isinstance(b, df.GenericVector):
            return NotImplemented
        x = self.A.create_vec(dim=1)

        if x.size() != b.size():
            raise RuntimeError(
                'incompatible dimensions for PETSc matvec, %d != %d'%(len(x),len(b)))
        b_, x_ = mat(self.A).createVecs()
        
        self.pc.apply(vec(b), vec(x))

        return x

    @vec_pool
    def create_vec(self, dim=1):
        return self.A.create_vec(dim)
