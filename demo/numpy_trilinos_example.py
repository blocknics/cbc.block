from block import *
from block.iterative import *
from block.algebraic.trilinos import *

import dolfin
import numpy 

def numpy2epetra(M):
    import numpy
    from PyTrilinos import Epetra

    arr = numpy.asarray(M)
    max_nnz = 1
    for row in arr:
      nnz = numpy.count_nonzero(row)
      if nnz > max_nnz:
        max_nnz = nnz

    comm = Epetra.MpiComm(Epetra.CommWorld)
    rowmap = Epetra.Map(arr.shape[0], 0, comm)
    colmap = Epetra.Map(arr.shape[1], 0, comm)
    mat = Epetra.CrsMatrix(Epetra.Copy, rowmap, max_nnz, True)
    for row in rowmap.MyGlobalElements():
      indices = numpy.array(numpy.where(arr[row])[0], dtype=numpy.intc)
      values = arr[row][indices]
      if mat.InsertGlobalValues(row, values, indices) != 0:
        raise RuntimeError('failed to insert row')
    if mat.FillComplete() != 0:
      raise RuntimeError('failed to FillComplete')

    return matrix_op(mat)

N = 120
A = numpy.zeros([N,N])
for i in range(N):
  A[i,i] = 2 
  if i > 0: 
    A[i-1,i] = -1 
  if i < N-1: 
    A[i+1,i] = -1 

A = numpy2epetra(A)

from numpy import random
x = dolfin.Vector(None, N)
x[:] = random.random(N)


Ainv = ConjGrad(A, precond=ML(A), tolerance=1e-10, show=3)

y = Ainv*x
print ("CG/ML converged in ", len(Ainv.residuals), " iterations "))





