from __future__ import division
from __future__ import print_function
from builtins import range
"""Base class for iterative solvers."""

from block.block_base import block_base
import dolfin as df
import os

class iterative(block_base):
    def __init__(self, A, precond=1.0, tolerance=1e-5, initial_guess=None,
                 iter=None, maxiter=200, name=None, show=1, rprecond=None,
                 nonconvergence_is_fatal=False, retain_guess=False, relativeconv=False,
                 callback=None, **kwargs):

        self.B = precond
        self.R = rprecond
        self.A = A
        self.initial_guess = initial_guess
        self.retain_guess = retain_guess
        self.nonconvergence_is_fatal = nonconvergence_is_fatal
        self.show = show
        self.callback = callback
        self.relativeconv = relativeconv
        self.kwargs = kwargs
        self.name = name if name else self.__class__.__name__
        if iter is not None:
            tolerance = 0
            maxiter = iter
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.cputime = 0.

    def create_vec(self, dim=1):
        return self.A.create_vec(dim)

    def matvec(self, b):
        from time import time
        from block.block_vec import block_vec
#        from dolfin import log, info, Progress
        TRACE = 13 # dolfin.TRACE

        T = time()

        # If x and initial_guess are block_vecs, some of the blocks may be
        # scalars (although they are normally converted to vectors by bc
        # application). To be sure, call allocate() on them.

        if isinstance(b, block_vec) and not b.allocated():
            # Create a shallow copy to call allocate() on, to avoid changing the caller's copy of b
            b = block_vec(len(b), b.blocks)
            b.allocate(self.A, dim=0)

        if self.initial_guess is not None:
            # Most (all?) solvers modify x, so make a copy to avoid changing
            # the caller's copy of x
            from block.block_util import copy
            x = copy(self.initial_guess)
            if isinstance(x, block_vec):
                x.allocate(self.A, dim=1)
        else:
            x = self.A.create_vec(dim=1)
            x.zero()

        try:
#            log(TRACE, self.__class__.__name__+' solve of '+str(self.A))
#            if self.B != 1.0:
#                log(TRACE, 'Using preconditioner: '+str(self.B))
#            progress = Progress(self.name, self.maxiter)
            if self.tolerance < 0:
                tolerance = -self.tolerance
                relative = True
            else:
                tolerance = self.tolerance
                relative = False
            progress = 0 
            x = self.method(self.B, self.AR, x, b, tolerance=tolerance,
                            relativeconv=self.relativeconv, maxiter=self.maxiter,
                            progress=progress, callback=self.callback,
                            **self.kwargs)
            del progress # trigger final printout
        except Exception:
#            from dolfin import warning
            print("Error solving " + self.name)
            raise
        x, self.residuals, self.alphas, self.betas = x

        if self.tolerance == 0:
            msg = "done"
        elif self.converged:
            msg = "converged"
        else:
            msg = "NOT CONV."

        cputime = time()-T

        comm = df.MPI.comm_world
        # Get the average
        self.cputime = comm.allreduce(cputime)/comm.size
        if self.show == 1:
            print('%s %s [iter=%2d, time=%.2fs, res=%.1e]' \
                % (self.name, msg, self.iterations, self.cputime, self.residuals[-1]))
        elif self.show >= 2:
            print('%s %s [iter=%2d, time=%.2fs, res=%.1e, true res=%.1e]' \
                % (self.name, msg, self.iterations, self.cputime, self.residuals[-1], (self.A*x-b).norm('l2')))
        if self.show == 3 and not "DOLFIN_NOPLOT" in os.environ:
            if comm.rank == 0:
                try:
                    from matplotlib import pyplot
                    pyplot.figure('%s convergence (show=3)'%self.name)
                    pyplot.semilogy(self.residuals)
                    pyplot.show(block=True)
                except:
                    pass

        if self.R is not None:
            x = self.R*x

        if self.retain_guess:
            self.initial_guess = x

        if True:
            # Tag vectors with iteration counts so that they can be picked up
            # by regression tests.
            x._regr_test_niter = self.iterations
            if isinstance(x, block_base):
                for block in x:
                    block._regr_test_niter = self.iterations

        if not self.converged and self.nonconvergence_is_fatal:
            raise RuntimeError('Not converged')

        return x

    def __call__(self, initial_guess=None, precond=None, tolerance=None,
                 iter=None, maxiter=None, name=None, show=None, rprecond=None,
                 nonconvergence_ok=None, callback=None):
        """Allow changing the parameters within an expression, e.g. x = Ainv(initial_guess=x) * b"""
        import copy
        ret = copy.copy(self) # shallow copy
        if precond       is not None: ret.B = precond
        if rprecond      is not None: ret.R = rprecond
        if initial_guess is not None: ret.initial_guess = initial_guess
        if nonconvergence_ok is not None: ret.nonconvergence_ok = nonconvergence_ok
        if show          is not None: ret.show = show
        if name          is not None: ret.name = name
        if tolerance     is not None: ret.tolerance = tolerance
        if maxiter       is not None: ret.maxiter = maxiter
        if callback      is not None: ret.callback = callback
        if iter is not None:
            ret.tolerance = 0
            ret.maxiter = iter
        return ret

    @property
    def AR(self):
        return self.A if self.R is None else self.A*self.R

    @property
    def iterations(self):
        return len(self.residuals)-1
    @property
    def converged(self):
        eff_tolerance = self.tolerance
        if eff_tolerance == 0:
            return True
        if eff_tolerance < 0:
            eff_tolerance *= -self.residuals[0]
        if self.relativeconv: 
            eff_tolerance *= self.residuals[0]
        return self.residuals[-1] <= eff_tolerance

    def eigenvalue_estimates(self):
        #####
        # Adapted from code supplied by KAM (Simula PyCC; GPL license),
        #####

        # eigenvalues estimates in terms of alphas and betas

        import numpy

        n = len(self.alphas)
        if n == 0:
            raise RuntimeError('Eigenvalues can not be estimated, no alphas/betas')
        M = numpy.zeros([n,n])
        M[0,0] = 1/self.alphas[0]
        for k in range(1, n):
            M[k,k] = 1/self.alphas[k] + self.betas[k-1]/self.alphas[k-1]
            M[k,k-1] = numpy.sqrt(self.betas[k-1])/self.alphas[k-1]
            M[k-1,k] = M[k,k-1]
        e,v = numpy.linalg.eig(M)
        e.sort()
        return e

    def __str__(self):
        return '<%d %s iterations on %s>'%(self.maxiter, self.name, self.A)

