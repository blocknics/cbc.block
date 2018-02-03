
import numpy 

from block import *
from block.iterative import *
from block.algebraic.petsc import *

import dolfin

class numpy_op(block_base):
    from block.object_pool import vec_pool
    import numpy

    def __init__(self, M):
        self.M = numpy.asmatrix(M)

    def matvec(self, b):
        from dolfin import GenericVector
        if not isinstance(b, GenericVector):
            return NotImplemented
        x = self.create_vec(dim=0)
        if len(b) != self.M.shape[1]:
            raise RuntimeError(
                'incompatible dimensions for %s matvec, %d != %d'%(self.__class__.__name__,self.M.shape[1],len(b)))
        y=self.M.dot(b.array())
        x[:] = numpy.asarray(y)[0]
        return x

    def transpmult(self, b):
      return numpy_op(self.M.T).matvec(b)

    @vec_pool
    def create_vec(self, dim=1):
        return dolfin.Vector(None, self.M.shape[dim])

    def __str__(self):
        format = '<%s %dx%d>'
        return format%(self.__class__.__name__, self.M.shape[0], self.M.shape[1])


N = 120
A = numpy.zeros([N,N])
for i in range(N):
  A[i,i] = 2 
  if i > 0: 
    A[i-1,i] = -1 
  if i < N-1: 
    A[i+1,i] = -1 


A = numpy_op(A)
x = dolfin.Vector(None, N)
from numpy import random
x[:] = random.random(N)

Ainv = ConjGrad(A, tolerance=1e-10, show=2)

y = Ainv*x

print ("CG converged in ", len(Ainv.residuals), " iterations ")




