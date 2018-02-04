"""This demo solves the Mixed formulation of Poisson problem equations with block preconditioning.


The algebraic system to be solved can be written as

  BB^ AA [sigma u]^T = BB^ [b 0]^T,

where AA is a 2x2 block system with zero in the (2,2) block

       | A   B |
  AA = |       |,
       | C   0 |

and BB^ approximates the inverse of the block operator

       | A   0 |
  BB = |       |.
       | 0   L |
"""

from dolfin import *
from block import *
from block.dolfin_util import *
from block.iterative import *
from block.algebraic.petsc import *

import os


N = 48 
mesh = UnitSquareMesh(N, N) 
# Define function spaces
V = FunctionSpace(mesh, "BDM", 1)
Q = FunctionSpace(mesh, "DG", 0)


# Define variational problem and assemble matrices
v, u = TestFunction(V), TrialFunction(V)
q, p = TestFunction(Q), TrialFunction(Q)

f = Constant((0, 0))

beta = Constant(0.5)

a11 = inner(v,u)*dx
a12 = div(v)*p*dx
a21 = div(u)*q*dx
p22 = beta*inner(jump(p),jump(q))*dS() + p*q*ds() #+ p*q*dx 
L1  = inner(v, f)*dx
L2  = inner(q, Constant(0))*dx

P22  = assemble(p22)

# Create the block matrix/vector, and apply boundary conditions. A diagonal
# matrix is automatically created to replace the (2,2) block in AA, since bc2
# makes the block non-zero.
AA = block_assemble([[a11, a12],
                     [a21,  0 ]])
bb  = block_assemble([L1, L2])


# Extract the individual submatrices
[[A, B],
 [C, _]] = AA

# Create preconditioners: An ML preconditioner for A, and the inverse diagonal
# of the mass matrix for the (2,2) block.
PP11 = InvDiag(A)
PP22 = ML(P22)

prec = block_mat([[PP11, 0],
                  [0, PP22]])

# Create the block inverse, using the preconditioned Minimum Residual method
# (suitable for symmetric indefinite problems).

xx = AA.create_vec() 
xx.randomize()

AAinv = MinRes(AA, precond=prec, initial_guess=xx, tolerance=1e-4, maxiter=500, show=2)

# Compute solution
u, p = AAinv * bb


