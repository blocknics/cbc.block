from block import *
from block.algebraic.hazmath import AMG
from block.algebraic.petsc import KSP
from block.dolfin_util import BoxBoundary
from block.iterative import Richardson
from dolfin import *

#########################
# Basic Poisson (poisson.py)
#########################

mesh = UnitSquareMesh(16,16)
V = FunctionSpace(mesh, "CG", 1)

f = Expression("sin(pi*x[0])", degree=2)
u, v = TrialFunction(V), TestFunction(V)

a = u*v*dx + dot(grad(u), grad(v))*dx
L = f*v*dx

A = assemble(a)
b = assemble(L)

u = Function(V)
solve(A, u.vector(), b)

#########################
# Solve as single-matrix
#########################

Ainv = KSP(A, precond=AMG(A), ksp_type='cg')
# Alternatively:
#Ainv = KSP(A, ksp_type='cg', pc_type='hypre', pc_hypre_type='boomeramg')
x = Ainv*b

check_expected('x', x, expected=u.vector(), show=True)

#########################
# Solve as (trivial) block-matrix
#########################

AA = block_mat([[A]])
bb = block_vec([b])
AAinv = KSP(AA, ksp_type='cg', pc_type='none')

xx = AAinv*bb # PETSC_ERR_ARG_WRONG    62   /* wrong argument (but object probably ok) */

check_expected('x', xx[0], expected=u2.vector(), show=True)

