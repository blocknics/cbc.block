"""Demo showing some of the basic block matrix transform functionality."""
from __future__ import print_function

from dolfin import *
from block import *
from block.iterative import *
import numpy as np

mesh = UnitSquareMesh(2,2)
V    = FunctionSpace(mesh, "CG", 1)
v, u = TestFunction(V), TrialFunction(V)

tensor = block_mat(np.array([[ 1, 0],
                             [-1, 1.]]))
print('Block matrix multiplication: composed object and its result')
I = block_simplify(block_mat([[1, 0],
                              [0, 1]]))
B = block_transpose(tensor)*(tensor+I)
print('\nI =',I)
print('\nB =',B)
print('\nBx=',block_collapse(B))

A = assemble(inner(grad(u),grad(v))*dx)
M = assemble(u*v*dx)
b = assemble(v*dx)
K = block_kronecker(A,tensor)

print('\n=================')
print('Block matrix multiplication: Kronecker product and its result')
print('\nK =', K)
print('\nKx=', block_collapse(K))

print('\n=================')
print('Block matrix multiplication: Inverse of the Kronecker product.')
print('This can be formed by explicit inversion of the tensor and an')
print('iterative solver, either blockwise or on the full system.')

import numpy
C,D = K

Di  = block_mat(numpy.linalg.inv(np.fromiter(tensor.blocks.flatten(),
                                             dtype=float).reshape(tensor.blocks.shape)))
print('\nKinv1=', block_kronecker(Di,ConjGrad(A)))
print('\nKinv2=', Di*ConjGrad(C))

print('\n=================')
print('Repeated solution of similar equations. This type of construct')
print('may turn up in the solution of inverse problems.')

AA  = block_mat.diag([0, M+A, -M], n=5)
AAp = AA.scheme('gs', inverse=ConjGrad, reverse=True)
bb = block_vec([b]*5)

xx = Richardson(AA, precond=AAp)*bb

# Or equivalently, using the Kronecker product. Note that block_collapse is
# required here to turn the product into a single block matrix. The
# Gauss-Seidel scheme is only defined on block_mat.

tensor1 = block_mat.diag(1, n=5)
tensor2 = block_mat.diag([0,1,-1], n=5)

AA = block_collapse(block_kronecker(A, tensor1) + block_kronecker(M, tensor2))
AAp = AA.scheme('gs', inverse=ConjGrad, reverse=True)

xx = Richardson(AA, precond=AAp)*bb
